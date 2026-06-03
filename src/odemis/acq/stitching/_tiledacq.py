# -*- coding: utf-8 -*-
"""
@author: Bassim Lazem

Copyright © 2020 Bassim Lazem, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the terms
of the GNU General Public License version 2 as published by the Free Software
Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.
"""
import copy
import logging
import math
import os
import statistics
import threading
import time
from abc import ABCMeta, abstractmethod
from concurrent.futures import CancelledError, ThreadPoolExecutor, TimeoutError
from concurrent.futures._base import CANCELLED, FINISHED, PENDING, RUNNING
from enum import Enum
from itertools import groupby
from typing import Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt

import numpy
import psutil
from scipy.interpolate import LinearNDInterpolator, NearestNDInterpolator
from scipy.ndimage import binary_fill_holes
from scipy.spatial import Delaunay
from shapely.geometry import MultiPoint, Point, Polygon, box

from odemis import dataio, model
from odemis.acq import acqmng
from odemis.acq.align.autofocus import MTD_EXHAUSTIVE, AutoFocus
from odemis.acq.align.roi_autofocus import (
    autofocus_in_roi,
    estimate_autofocus_in_roi_time,
)
from odemis.acq.stitching._constants import (
    REGISTER_GLOBAL_SHIFT,
    REGISTER_IDENTITY,
    WEAVER_MEAN,
)
from odemis.acq.stitching._simple import register, weave
from odemis.acq.stream import (
    ARStream,
    CLStream,
    EMStream,
    FluoStream,
    MultipleDetectorStream,
    SpectrumStream,
    Stream,
    executeAsyncTask,
    util,
)
from odemis.model import CancellableThreadPoolExecutor, DataArray
from odemis.util import dataio as udataio
from odemis.util import img, linalg, rect_intersect
from odemis.util.focus import MeasureOpticalFocus
from odemis.util.img import assembleZCube
from odemis.util.linalg import generate_triangulation_points
from odemis.util.raster import get_polygon_grid_cells, point_in_polygon

try:
    from sparc_calibrations.parabolic_mirror_alignment import (
        AlignmentAxis,
        ParabolicMirrorAlignmentTask,
    )
    sparc_calib_available = True
except ImportError as err:
    sparc_calib_available = False
    logging.info("sparc_calibrations module not available, mirror z alignment not possible: %s", err)


# TODO: Find a value that works fine with common cases
# Ratio of the allowed difference of tile focus from good focus
FOCUS_FIDELITY = 0.3
# Limit focus range, half the margin will be used on each side of initial focus
FOCUS_RANGE_MARGIN = 100e-6  # m
# Indicate the number of tiles to skip during focus adjustment
SKIP_TILES = 3
# Starting index for the tile acquisition
# Start with a non-existing tile index so that the first tile is always moved to and acquired
START_INDEX = (-1, -1)

MOVE_SPEED_DEFAULT = 100e-6  # m/s
# Default range for the optical focus adjustment
SAFE_REL_RANGE_DEFAULT = (-50e-6, 50e-6)  # m
# Maximum distance is used to separate two focus points in overview acquisition using autofocus
# Increasing this distance may result in out of focus overview acquisition image
MAX_DISTANCE_FOCUS_POINTS = 450e-06  # in m

DEFAULT_FOV = (100e-6, 100e-6) # m
STITCH_SPEED = 1e8  # px/s

class FocusingMethod(Enum):
    NONE = 0  # Never auto-focus
    ALWAYS = 1  # Before every tile
    # If the previous tile focus level is too far from the original level
    ON_LOW_FOCUS_LEVEL = 2
    # Acquisition is done at several zlevels, and they are merged to obtain a focused image
    MAX_INTENSITY_PROJECTION = 3


def create_z_interpolator(points: List[Tuple[float, float]], z_values: List[float])-> Optional[Callable]:
    """
    Create a Z-height interpolator from survey points.

    Uses scipy LinearNDInterpolator for smooth interpolation within the convex hull
    of survey points, and NearestNDInterpolator as fallback for extrapolation beyond.

    :param points: List of (x, y) tuples representing survey point positions
    :param z_values: List of z values corresponding to each survey point
    :return: Interpolator function (x, y) -> z, or None if interpolation fails
    """
    if not points or not z_values or len(points) != len(z_values):
        logging.error(f"Cannot create Z interpolator")
        return None

    try:
        # Convert to numpy arrays
        points = numpy.array(points)
        z_values = numpy.array(z_values)

        # Create linear interpolator for smooth interpolation
        lin_interp = LinearNDInterpolator(points, z_values, fill_value=numpy.nan)

        # Create nearest-neighbor interpolator for fallback extrapolation
        nn_interp = NearestNDInterpolator(points, z_values)

        # Combined interpolator: use linear inside, nearest outside
        def interpolator(x, y):
            """Interpolate Z at given (x, y) position."""
            z_lin = lin_interp(x, y)

            # If linear interpolation returns NaN (outside convex hull),
            # fall back to nearest-neighbor extrapolation
            if numpy.isnan(z_lin):
                z_lin = float(nn_interp(x, y))
            else:
                z_lin = float(z_lin)

            return z_lin

        logging.info(
            f"Created Z-height interpolator from {len(points)} survey points"
        )
        return interpolator

    except Exception:
        logging.exception("Failed to create Z interpolator")
        return None


def order_positions_spiral_outward(positions: List[Tuple[float, float]],
                                   current_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Order positions in concentric rings spiraling inward to center, then outward.

    Optimal for:
    - Circular distributions (wafer area)
    - Polygonal boundaries
    - Any convex region with interior starting point

    Algorithm:
    1. Calculate centroid of all positions (center point)
    2. Find the position CLOSEST to current stage position (critical for collision avoidance!)
    3. Compute distance and angle from centroid for each position
    4. Group into concentric rings by distance from centroid
    5. Sort each ring by angle (polar coordinates)
    6. Start spiral from closest position's ring
    7. Traverse INWARD to innermost ring (decreasing distance)
    8. Then traverse OUTWARD to outermost ring (increasing distance)
    9. Alternate direction within each ring (prevents backtracking)

    Result: Smooth concentric spiral with NO backtracking, NO big jumps.

    Visual Flow (closest point at Ring 2):

    Start at Ring 2 (closest to stage):
    Ring 2: ★→●→●→● ↘
                        ↘
    Spiral INWARD:
    Ring 1:          ●←●←●
                    ↗
                ↗
    Ring 0: ●→●→● (center)

    Then spiral OUTWARD:
    Ring 2: ●→●→● (already visited, skip)
    Ring 3: ●←●←●
    Ring 4: ●→●→●

    :param positions: List of (x, y) tuples
    :param current_pos: Current stage position (used to find spiral start point)
    :return: List of positions ordered in spiral pattern starting from closest point
    """

    positions_array = numpy.array(positions)

    # Find centroid of all positions (center reference point)
    centroid = positions_array.mean(axis=0)

    # ========== FIND CLOSEST POSITION TO CURRENT POS (KEY FOR SAFETY) ==========
    closest_distance_sq = float('inf')
    closest_pos = None
    closest_idx = None

    for idx, pos in enumerate(positions):
        distance_sq = (pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2
        if distance_sq < closest_distance_sq:
            closest_distance_sq = distance_sq
            closest_pos = pos
            closest_idx = idx

    logging.debug(
        "Closest position to current_pos %s @%d: %s at distance %.2e m",
        current_pos, closest_idx, closest_pos, numpy.sqrt(closest_distance_sq),
    )

    # Compute distance and angle from centroid (for ring grouping and spiral ordering)
    vectors = positions_array - centroid
    distances = numpy.linalg.norm(vectors, axis=1)
    angles = numpy.arctan2(vectors[:, 1], vectors[:, 0])

    # Determine ring spacing: adaptive based on point density
    distance_range = distances.max() - distances.min()
    num_rings = max(2, min(10, len(positions) // 5))  # 2-10 rings based on point count
    ring_spacing = distance_range / num_rings if distance_range > 0 else 1.0

    # Group points into concentric rings
    # Each ring entry stores (angle, position, original_index) so we can
    # identify the closest position later without fragile float comparisons.
    rings = {}
    closest_ring_key = None

    for idx, (pos, dist, angle) in enumerate(zip(positions, distances, angles)):
        # Determine which ring this point belongs to
        if ring_spacing > 0:
            ring_key = int(round((dist - distances.min()) / ring_spacing))
        else:
            ring_key = 0

        if ring_key not in rings:
            rings[ring_key] = []
        rings[ring_key].append((angle, list(pos), idx))

        # Track which ring contains the closest position (by index, not float comparison)
        if idx == closest_idx:
            closest_ring_key = ring_key

    if closest_ring_key is None:
        logging.error("Could not determine ring for closest position, defaulting to ring 0")
        closest_ring_key = 0

    # ========== BUILD SPIRAL: INWARD THEN OUTWARD ==========
    ordered_path = []

    # PHASE 1: Traverse INWARD from closest ring to center (ring 0)
    logging.debug("Closest position is in ring %d. Spiraling inward to center (ring 0)", closest_ring_key)

    reverse_direction = False

    # Go from closest_ring_key down to 0 (innermost)
    for ring_idx in range(closest_ring_key, -1, -1):
        if ring_idx not in rings:
            continue

        ring_positions = rings[ring_idx]
        ring_positions.sort(key=lambda x: x[0])  # Sort by angle

        # For the closest ring, put the closest position first
        if ring_idx == closest_ring_key:
            # Find closest position in ring by original index
            closest_idx_in_ring = None
            for i, (_, _, orig_idx) in enumerate(ring_positions):
                if orig_idx == closest_idx:
                    closest_idx_in_ring = i
                    break

            if closest_idx_in_ring is not None:
                # Determine initial direction (forward or backward from closest)
                # Wrap around so that ALL positions in the ring are visited.
                if closest_idx_in_ring < len(ring_positions) / 2:
                    # Closer to start of ring, traverse forward (clockwise), then wrap
                    wrapped = ring_positions[closest_idx_in_ring:] + ring_positions[:closest_idx_in_ring]
                    ordered_path.extend([tuple(pos) for _, pos, _ in wrapped])
                    reverse_direction = True  # Next ring (inward) goes backward (counter-clockwise)
                else:
                    # Closer to end of ring, traverse backward (counter-clockwise), then wrap
                    back_half = list(reversed(ring_positions[:closest_idx_in_ring + 1]))
                    front_half = list(reversed(ring_positions[closest_idx_in_ring + 1:]))
                    ordered_path.extend([tuple(pos) for _, pos, _ in back_half + front_half])
                    reverse_direction = False  # Next ring (inward) goes forward (clockwise)

                logging.debug(
                    "Ring %d (closest): Starting from index %d, "
                    "direction=%s, positions: %d",
                    ring_idx, closest_idx_in_ring,
                    'backward' if reverse_direction else 'forward',
                    len(ring_positions),
                )
            else:
                # Fallback: could not locate closest in ring (should not happen)
                ordered_path.extend([tuple(pos) for _, pos, _ in ring_positions])
                reverse_direction = not reverse_direction
        else:
            # For inward rings, alternate direction (smooth spiral)
            if reverse_direction:
                ordered_path.extend([tuple(pos) for _, pos, _ in reversed(ring_positions)])
                logging.debug("Ring %d (inward): backward, positions: %d", ring_idx, len(ring_positions))
            else:
                ordered_path.extend([tuple(pos) for _, pos, _ in ring_positions])
                logging.debug("Ring %d (inward): forward, positions: %d", ring_idx, len(ring_positions))
            reverse_direction = not reverse_direction

    # PHASE 2: Traverse OUTWARD from closest ring to outermost
    logging.debug("Now spiraling outward from ring %d to outermost", closest_ring_key)

    max_ring = max(rings.keys()) if rings else 0

    # Go from closest_ring_key+1 up to max_ring (outermost)
    for ring_idx in range(closest_ring_key + 1, max_ring + 1):
        if ring_idx not in rings:
            continue

        ring_positions = rings[ring_idx]
        ring_positions.sort(key=lambda x: x[0])  # Sort by angle

        # Alternate direction in successive rings (spiral effect, no backtracking)
        if reverse_direction:
            ordered_path.extend([tuple(pos) for _, pos, _ in reversed(ring_positions)])
            logging.debug("Ring %d (outward): backward, positions: %d", ring_idx, len(ring_positions))
        else:
            ordered_path.extend([tuple(pos) for _, pos, _ in ring_positions])
            logging.debug("Ring %d (outward): forward, positions: %d", ring_idx, len(ring_positions))
        reverse_direction = not reverse_direction

    logging.info(
        "Spiral pattern (inward->outward): %d rings, %d total positions. "
        "Starting ring: %d, Center (ring 0), Outermost (ring %d), "
        "Centroid: (%.2e, %.2e), Distance range: %.2e m",
        len(rings), len(ordered_path), closest_ring_key, max_ring,
        centroid[0], centroid[1], distance_range,
    )

    return ordered_path


def order_positions_snake_lanes(positions: List[Tuple[float, float]],
                                current_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Order positions using snake/boustrophedon pattern with smooth lane transitions.

    Optimal for:
    - Grid-like distributions
    - Starting point outside the survey region
    - Rectangular/square survey areas

    Algorithm:
    1. Analyze grid structure: determine dominant axis (X or Y)
    2. Create lanes perpendicular to dominant axis
    3. Find closest position to current stage
    4. Traverse lanes in snake pattern (alternating directions)
    5. Connect lanes smoothly to nearest endpoint (no big jumps)

    Result: Parallel passes through survey area with minimal backtracking.

    Visual example:
        Lane 0: ★→●→●→● ↘
                            ↘
        Lane 1:          ●←●←●
                        ↗
                    ↗
        Lane 2: ●→●→● →

    :param positions: List of (x, y) tuples
    :param current_pos: Current stage position
    :return: List of positions ordered in snake pattern
    """
    if not positions:
        return []

    # ========== PHASE 1: ANALYZE GRID STRUCTURE ==========
    xs = [p[0] for p in positions]
    ys = [p[1] for p in positions]

    x_range = max(xs) - min(xs)
    y_range = max(ys) - min(ys)

    # Determine dominant axis (where points are more spread out)
    use_y_lanes = x_range >= y_range  # If X spread > Y spread, create Y-lanes
    lane_threshold = 10.e-6  # 10 micrometers: points within this belong to same lane

    if use_y_lanes:
        # Create lanes based on X coordinate (lanes run along Y direction)
        sorted_positions = sorted(positions, key=lambda p: p[0])
        lanes_dict = {}
        lane_key = 0
        last_x = sorted_positions[0][0]

        for pos in sorted_positions:
            # New lane if X coordinate jumps more than threshold
            if abs(pos[0] - last_x) > lane_threshold:
                lane_key += 1
            if lane_key not in lanes_dict:
                lanes_dict[lane_key] = []
            lanes_dict[lane_key].append(pos)
            last_x = pos[0]

        # Sort each lane by Y coordinate (ascending)
        for lane_positions in lanes_dict.values():
            lane_positions.sort(key=lambda p: p[1])

        lanes = lanes_dict
        lane_keys = sorted(lanes_dict.keys())
        axis_name = "Y"

    else:
        # Create lanes based on Y coordinate (lanes run along X direction)
        sorted_positions = sorted(positions, key=lambda p: p[1])
        lanes_dict = {}
        lane_key = 0
        last_y = sorted_positions[0][1]

        for pos in sorted_positions:
            # New lane if Y coordinate jumps more than threshold
            if abs(pos[1] - last_y) > lane_threshold:
                lane_key += 1
            if lane_key not in lanes_dict:
                lanes_dict[lane_key] = []
            lanes_dict[lane_key].append(pos)
            last_y = pos[1]

        # Sort each lane by X coordinate (ascending)
        for lane_positions in lanes_dict.values():
            lane_positions.sort(key=lambda p: p[0])

        lanes = lanes_dict
        lane_keys = sorted(lanes_dict.keys())
        axis_name = "X"

    # ========== PHASE 2: FIND STARTING POINT ==========
    closest_distance_sq = float('inf')
    closest_pos = None
    closest_lane_key = None

    for lane_key in lane_keys:
        lane_positions = lanes[lane_key]
        for pos in lane_positions:
            distance_sq = (pos[0] - current_pos[0])**2 + (pos[1] - current_pos[1])**2
            if distance_sq < closest_distance_sq:
                closest_distance_sq = distance_sq
                closest_pos = pos
                closest_lane_key = lane_key

    # ========== PHASE 3: BUILD SNAKE PATTERN WITH SMOOTH LANE TRANSITIONS ==========
    ordered_path = []
    visited_lanes = set()
    current_lane = closest_lane_key
    reverse_direction = False

    while len(visited_lanes) < len(lane_keys):
        lane_positions = lanes[current_lane][:]  # Copy to avoid modifications
        visited_lanes.add(current_lane)

        # For starting lane, begin from closest position
        if current_lane == closest_lane_key:
            try:
                start_idx = lane_positions.index(closest_pos)
                # Determine direction: closer to lane start or end?
                if start_idx < len(lane_positions) / 2:
                    # Closer to start: go forward to end, then wrap back for skipped positions
                    forward = lane_positions[start_idx:]
                    wrap_back = list(reversed(lane_positions[:start_idx]))
                    ordered_path.extend(forward + wrap_back)
                    reverse_direction = True  # Next lane goes backward
                else:
                    # Closer to end: go backward to start, then wrap for remaining positions
                    backward = list(reversed(lane_positions[:start_idx + 1]))
                    wrap_forward = lane_positions[start_idx + 1:]
                    ordered_path.extend(backward + wrap_forward)
                    reverse_direction = False  # Next lane goes forward
            except ValueError:
                ordered_path.extend(lane_positions)
                reverse_direction = not reverse_direction
        else:
            # For subsequent lanes, alternate direction (snake pattern)
            if reverse_direction:
                ordered_path.extend(reversed(lane_positions))
            else:
                ordered_path.extend(lane_positions)
            reverse_direction = not reverse_direction

        # Find next lane: closest unvisited lane to current endpoint
        if ordered_path and len(visited_lanes) < len(lane_keys):
            last_pos = ordered_path[-1]

            # Find unvisited lane with nearest endpoint
            closest_next_lane = None
            closest_next_distance = float('inf')

            for lane_key in lane_keys:
                if lane_key not in visited_lanes:
                    next_lane_positions = lanes[lane_key]

                    # Calculate distance to both ends of this lane
                    # Choose the end that is closer (will be traversal start)
                    dist_to_first = (next_lane_positions[0][0] - last_pos[0])**2 + \
                                (next_lane_positions[0][1] - last_pos[1])**2
                    dist_to_last = (next_lane_positions[-1][0] - last_pos[0])**2 + \
                                (next_lane_positions[-1][1] - last_pos[1])**2

                    closest_end_distance = min(dist_to_first, dist_to_last)

                    if closest_end_distance < closest_next_distance:
                        closest_next_distance = closest_end_distance
                        closest_next_lane = lane_key

            if closest_next_lane is not None:
                current_lane = closest_next_lane
            else:
                break

    logging.info(
        f"Snake pattern: {len(lanes)} lanes along {axis_name}-direction, "
        f"{len(ordered_path)} total positions. "
        f"Range X: {x_range:.2e} m, Range Y: {y_range:.2e} m"
    )

    return ordered_path


def order_positions(positions: List[Tuple[float, float]],
                    current_pos: Tuple[float, float]) -> List[Tuple[float, float]]:
    """
    Order positions using intelligent adaptive path planning.

    HYBRID ALGORITHM:
    1. Detect if current_pos is inside the convex hull of positions
    2. If INSIDE: Use spiral/concentric ring traversal (optimal for circles, polygons)
    3. If OUTSIDE: Use snake/lane pattern (optimal for grids, peripheral points)

    This unified approach handles arbitrary point distributions with minimal stage travel.

    INSIDE CASE - Spiral from center outward:
        Ring 3: ●←●←●  ↙
                ↙
        Ring 2: ●→●→●  ↘
                ↘
        Ring 1:    ★  ↘ (start)
                ↗
        Ring 0: ●→●→●

    OUTSIDE CASE - Snake lanes:
        Lane 0: ★→●→●→● ↘
                        ↘
        Lane 1:          ●←●←●
                        ↗
                    ↗
        Lane 2: ●→●→● →

    :param positions: List of (x, y) tuples representing all points to visit
    :param current_pos: Current stage position (x, y) tuple
    :return: List of positions ordered in optimized path (spiral or snake)
    """
    if not positions:
        return []
    if len(positions) == 1:
        return positions
    if len(positions) == 2:
        dist_0 = (positions[0][0] - current_pos[0])**2 + (positions[0][1] - current_pos[1])**2
        dist_1 = (positions[1][0] - current_pos[0])**2 + (positions[1][1] - current_pos[1])**2
        return positions if dist_0 <= dist_1 else [positions[1], positions[0]]

    # ========== DETECT IF CURRENT POS IS INSIDE CONVEX HULL ==========

    is_inside = False
    try:
        tri = Delaunay(positions)
        is_inside = tri.find_simplex(current_pos) >= 0
        logging.debug(f"Current position {'INSIDE' if is_inside else 'OUTSIDE'} convex hull")
    except Exception as e:
        logging.debug(f"Could not compute convex hull (likely collinear points): {e}. Using snake pattern.")
        is_inside = False

    if is_inside:
        # ========== SPIRAL PATTERN FROM CENTER OUTWARD ==========
        logging.info(f"Current position inside convex hull - using spiral outward pattern for {len(positions)} positions")
        return order_positions_spiral_outward(positions, current_pos)
    else:
        # ========== SNAKE LANE PATTERN FROM OUTSIDE ==========
        logging.info(f"Current position outside convex hull - using snake lane pattern for {len(positions)} positions")
        return order_positions_snake_lanes(positions, current_pos)


class BaseTiledAcquisitionTask(metaclass=ABCMeta):

    def __init__(self, registrar=REGISTER_GLOBAL_SHIFT, weaver=WEAVER_MEAN):
        """
        :param registrar: (REGISTER_* or None) type of registration method. If registrar and weaver are None, do not stitch.
        :param weaver: (WEAVER_* or None) type of weaving method. If registrar and weaver are None, do not stitch.
        """
        self._registrar = registrar
        self._weaver = weaver

    @staticmethod
    def getFov(sd):
        """
        sd (Stream or DataArray): If it's a stream, it must be a live stream,
          and the FoV will be estimated based on the settings.
        return (float, float): width, height in m
        """
        if isinstance(sd, model.DataArray):
            # The actual FoV, as the data recorded it
            im_bbox = img.getBoundingBox(sd)
            logging.debug("Bounding box of stream data: %s", im_bbox)
            return im_bbox[2] - im_bbox[0], im_bbox[3] - im_bbox[1]
        elif isinstance(sd, Stream):
            return get_fov(sd)
        else:
            raise TypeError("Unsupported object")

    @classmethod
    def guessSmallestFov(cls, ss):
        """
        Return (float, float): smallest width and smallest height of all the FoV
          Note: they are not necessarily from the same FoV.
        raise ValueError: If no stream selected
        """
        fovs = [cls.getFov(s) for s in ss]
        if not fovs:
            raise ValueError("No stream so no FoV, so no minimum one")

        return (min(f[0] for f in fovs),
                min(f[1] for f in fovs))

    def _stitchTiles(self, da_list):
        """
        Stitch the acquired tiles to create a complete view of the required total area
        :return: (list of DataArrays): a stitched data for each stream acquisition
        """
        st_data = []
        logging.info("Computing big image out of %d images", len(da_list))

        # TODO: Do this registration step in a separate thread while acquiring
        try:
            das_registered = register(da_list, method=self._registrar)
        except ValueError as exp:
            logging.warning("Registration with %s failed %s. Retrying with identity registrar.", self._registrar, exp)
            das_registered = register(da_list, method=REGISTER_IDENTITY)

        logging.info("Using weaving method %s.", self._weaver)
        # Weave every stream
        if isinstance(das_registered[0], tuple):
            for s in range(len(das_registered[0])):
                streams = []
                for da in das_registered:
                    streams.append(da[s])
                da = weave(streams, self._weaver)
                st_data.append(da)
        else:
            da = weave(das_registered, self._weaver)
            st_data.append(da)
        return st_data

    @abstractmethod
    def run(self):
        pass


class BaseTileAcquisitionController(metaclass=ABCMeta):
    """
    Controls the tile-by-tile acquisition loop.
    Subclasses define HOW to move, acquire, and order positions.
    """

    def __init__(self, streams, stage):
        self._streams = streams
        self._stage = stage

    @abstractmethod
    def estimateMemory(self):
        pass

    @abstractmethod
    def estimateTime(self, remaining=None):
        pass

    @abstractmethod
    def get_ordered_positions(self):
        """Return iterable of positions (type depends on subclass)."""
        pass

    @abstractmethod
    def move_to_position(self, position, prev_position, index):
        """Move stage to the next tile position."""
        pass

    @abstractmethod
    def acquire_at_position(self, position, index):
        """Acquire data at current position. Returns list of DataArrays."""
        pass

    def pre_acquire(self, position, index):
        """Hook called after move, before acquire (e.g., refocus). Override as needed."""
        pass

    def post_acquire(self, das, position, index):
        """Hook called after acquire (e.g., adjust focus). Override to transform das."""
        pass

    @abstractmethod
    def acquire_all(self):
        pass


class GridTiledAcquisitionController(BaseTileAcquisitionController):

    MEMPP = 22  # bytes per pixel, found empirically

    def __init__(self, streams, region, stage, overlap, sfov, future=None, settings_obs=None, log_path=None, centered_acq=True):
        super().__init__(streams, stage)
        self._polygon = self._convert_region_to_polygon(region)
        xmin, ymin, xmax, ymax = self._polygon.bounds
        self._area_size = (xmax - xmin, ymax - ymin)
        self._overlap = overlap
        self._sfov = sfov
        self._centered_acq = centered_acq
        self._future = future
        self._settings_obs = settings_obs
        self._log_path = log_path
        # save actual time take for tile acquisition, stage movement from one tile to another
        # and stitching time for each tile. The keys tell the action, i.e. move or stitch and the time
        # taken for the same action per each tile is saved as a list of floats
        self._save_time: Dict[str, List[float]] = {}
        if self._future is not None:
            self._future.running_subf: future.Future = model.InstantaneousFuture()
            self._future._task_lock = threading.Lock()
            self._future._task_state = PENDING
        self._exporter = None
        self._fn_bs = None
        self._fn_ext = None
        self._log_dir = None
        self._save_executor = None
        self.average_acquisition_time = None
        
        # Rough estimate of the stage movement speed, for estimating the extra
        # duration due to movements
        self._move_speed = MOVE_SPEED_DEFAULT
        if model.hasVA(stage, "speed"):
            try:
                self._move_speed = (stage.speed.value["x"] + stage.speed.value["y"]) / 2
            except Exception as ex:
                logging.warning("Failed to read the stage speed: %s", ex)

        if self._log_path:
            filename = os.path.basename(self._log_path)
            if not filename:
                raise ValueError("Filename is not found on log path.")
            self._exporter = dataio.find_fittest_converter(filename)
            self._fn_bs, self._fn_ext = udataio.splitext(filename)
            self._log_dir = os.path.dirname(self._log_path)
            self._save_executor = ThreadPoolExecutor(max_workers=5)

        self._starting_pos, self._tile_indices = self._get_tile_coverage()
        self._number_of_tiles = len(self._tile_indices)
        logging.debug("Calculated number of tiles %s", self._number_of_tiles)

    def _updateFov(self, das, sfov):
        """
        Checks the fov and update it based on the data arrays
        das: list of DataArryas
        sfov: previous estimate for the fov
        :returns same fov or updated from the data arrays
        """
        afovs = [BaseTiledAcquisitionTask.getFov(d) for d in das]
        asfov = (min(f[0] for f in afovs),
                 min(f[1] for f in afovs))
        if not all(util.almost_equal(e, a) for e, a in zip(sfov, asfov)):
            logging.warning("Unexpected min FoV = %s, instead of %s", asfov, sfov)
            sfov = asfov
        return sfov

    def _convert_region_to_polygon(
            self,
            region: Union[Tuple[float, float, float, float], List[Tuple[float, float]]]
        ) -> Polygon:
        """
        Convert the region input into a polygon.
        :param region: Either a bounding box (xmin, ymin, xmax, ymax) or a list of (x, y) points.
        :return: A polygon representing the region.
        :raise: ValueError if the region does not form a valid polygon.
        :raise: ValueError if the region is not a bounding box or a list of points.
        """
        if isinstance(region, tuple) and len(region) == 4:
            # Normalize the bounding box to ensure the order is correct
            xmin, ymin, xmax, ymax = util.normalize_rect(region)
            if region[0] != xmin or region[1] != ymin:
                logging.warning("Acquisition area %s rearranged into %s", region, (xmin, ymin, xmax, ymax))
            points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]
        elif isinstance(region, list) and all(isinstance(point, tuple) and len(point) == 2 for point in region):
            points = region
        else:
            raise ValueError(
                "Region must either be a bounding box (xmin, ymin, xmax, ymax) or a list of (x, y) points."
            )

        polygon = Polygon(points)
        if not polygon.is_valid:
            raise ValueError("The provided region does not form a valid polygon.")
        return polygon

    def _get_tile_coverage(self) -> Tuple[Dict[str, float], List[Tuple[int, int]]]:
        """
        Calculate the exact tiles required to cover the region.
        return:
            starting_position: center of the first tile (at the top-left)
            tile_indices: List of (col, row) indices for the tiles
        """
        if self._polygon is None:
            raise ValueError("Polygon shape is not defined.")

        # The size of the smallest tile, non-including the overlap, which will be
        # lost (and also indirectly represents the precision of the stage)
        reliable_fov = ((1 - self._overlap) * self._sfov[0], (1 - self._overlap) * self._sfov[1])

        # Round up the number of tiles needed. With a twist: if we'd need less
        # than 1% of a tile extra, round down. This handles floating point
        # errors and other manual rounding when the requested area size is
        # exactly a multiple of the FoV.
        area_size = [(s - f * 0.01) if s > f else s
                     for s, f in zip(self._area_size, reliable_fov)]
        nx = math.ceil(area_size[0] / reliable_fov[0])
        ny = math.ceil(area_size[1] / reliable_fov[1])

        # If we have a little bit more tiles than needed, we have two choices
        # on how to spread them:
        # 1. Increase the total area acquired (and keep the overlap)
        # 2. Increase the overlap (and keep the total area)
        # We pick alternative 1 (no real reason)
        xmin, ymin, xmax, ymax = self._polygon.bounds

        if self._centered_acq:
            # Ensure extra acquisition area is symmetric around the center of the acquisition
            center = ((xmin + xmax) / 2, (ymin + ymax) / 2)
            total_size = (
                nx * reliable_fov[0] + self._sfov[0] * self._overlap,
                ny * reliable_fov[1] + self._sfov[1] * self._overlap,
            )
            xmin = center[0] - total_size[0] / 2
            ymax = center[1] + total_size[1] / 2

        # Create an empty grid for storing intersected tiles
        # An intersected tile is any tile in the grid that intersects with (or falls within) the given polygon
        tile_grid = numpy.zeros((ny, nx), dtype=bool)

        # Vectorized conversion of polygon points to grid coordinates
        points = numpy.array(self._polygon.exterior.coords)
        rows = numpy.floor((ymax - points[:, 1]) / reliable_fov[1]).astype(int)
        cols = numpy.floor((points[:, 0] - xmin) / reliable_fov[0]).astype(int)

        # Create array of (row, col) vertices
        polygon_vertices = numpy.stack((rows, cols), axis=1)

        intersected_tiles = get_polygon_grid_cells(polygon_vertices, include_neighbours=True)

        # The intersected tiles contains tiles which can be inside the polygon, outside the polygon and tiles which
        # actually intersect with the polygon. We only need the tiles which actually intersect the polygon, to binary
        # fill the holes and get the tile indices.
        for row, col in intersected_tiles:
            if not (0 <= row < ny and 0 <= col < nx):  # Certainly out of the polygon => not worthy to acquire
                continue
            # Define the bounds of the current tile
            tile_bounds = box(
                xmin + col * reliable_fov[0],
                ymax - (row + 1) * reliable_fov[1],
                xmin + (col + 1) * reliable_fov[0],
                ymax - row * reliable_fov[1]
            )
            # Check if the tile intersects with the polygon shape
            if self._polygon.intersects(tile_bounds):
                tile_grid[row, col] = True

        # Fill any holes in the grid to get contiguous tiles
        filled_grid = binary_fill_holes(tile_grid)
        rows, cols = numpy.nonzero(filled_grid)
        tile_indices = list(zip(cols.tolist(), rows.tolist()))

        # Calculate the starting position (top-left of the grid)
        if self._centered_acq:
            starting_position = {
                "x": xmin + reliable_fov[0] / 2,
                "y": ymax - reliable_fov[1] / 2
            }
        else:
            starting_position = {
                "x": xmin + self._sfov[0] / 2,
                "y": ymax - self._sfov[1] / 2
            }

        return starting_position, tile_indices

    def _sort_tile_indices_zigzag(self, tile_indices: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Sort the given tile indices in a zigzag scanning order.
        # Below is an example for a 5x3 tile grid:
        # A-->-->-->--v
        #             |
        # v--<--<--<---
        # |
        # --->-->-->--Z
        :param tile_indices: List of (int, int) tuples representing the tile indices
        :return: List of (int, int) tuples sorted in a zigzag scanning order
        """
        # Sort tile_indices by row first, then column
        tile_indices.sort(key=lambda t: (t[1], t[0]))

        # Group indices by rows and apply zigzag pattern
        sorted_indices = []
        for row, group in groupby(tile_indices, key=lambda t: t[1]):
            group = list(group)
            # Reverse the order of the group if the row is odd
            if row % 2 == 1:
                group.reverse()
            sorted_indices.extend(group)

        return sorted_indices

    def _moveToTile(self, idx, prev_idx, tile_size):
        """
        Move the stage to the tile position
        :param idx: (tuple (float, float)) current index of tile
        :param prev_idx: (tuple (float, float)) previous index of tile
        :param tile_size: (tuple (float, float)) total tile size
        """
        overlap = 1 - self._overlap
        # don't move on the axis that is not supposed to have changed
        m = {}
        idx_change = numpy.subtract(idx, prev_idx)
        if idx[0] != prev_idx[0]:  # x-axis changed
            m["x"] = self._starting_pos["x"] + idx[0] * tile_size[0] * overlap
        if idx[1] != prev_idx[1]:  # y-axis changed
            m["y"] = self._starting_pos["y"] - idx[1] * tile_size[1] * overlap

        # Compute the time to wait
        if prev_idx == START_INDEX:
            # If this is the first tile, wait for a long time to allow the stage to move
            # This is needed because the current stage position may be far from the first tile
            # so it may take a long time to move there
            timeout = 100  # s
        else:
            # For any tile after the first, don't wait forever for the stage to move,
            # guess the time it should take and then give a large margin
            t = math.hypot(abs(idx_change[0]) * tile_size[0] * overlap,
                           abs(idx_change[1]) * tile_size[1] * overlap) / self._move_speed
            timeout = 5 * t + 3  # s

        logging.debug("Moving to tile %s at %s m", idx, m)
        for i in range(3):  # Try moving up to 3 times
            if self._future._task_state == CANCELLED:
                raise CancelledError()
            self._future.running_subf = self._stage.moveAbs(m)
            try:
                self._future.running_subf.result(timeout)
            except ValueError:  # Typically, asked to move to the wrong place, let's give up
                raise
            except CancelledError:
                logging.warning("Move to tile %s cancelled", idx)
                raise
            except TimeoutError:
                self._future.running_subf.cancel()
                pos = self._stage.position.value
                pos_xy = (pos["x"], pos["y"])
                logging.warning("Failed to move to tile %s within %s s, now at %s%s",
                                idx, timeout, pos_xy, ", will retry" if i < 2 else "")
                time.sleep(0.1)
                continue
            except Exception as ex:
                pos = self._stage.position.value
                pos_xy = (pos["x"], pos["y"])
                logging.warning("Failed to move to tile %s, now at %s: %s%s",
                                idx, pos_xy, ex, ", will retry" if i < 2 else "")
                time.sleep(0.1)
                continue

            # The moved finished successfully, just stop trying to move, we are done!
            break
        else:
            # We exhausted all retries, we failed to move to the tile, so we cancel the acquisition
            logging.error("Failed to move to tile %s after 3 trials", idx)
            raise OSError(f"Failed to move to tile {idx} after 3 trials")

    def _acquireStreamTile(self, i, ix, iy, stream):
        """
        Calls acquire function and blocks until the data is returned.
        :return DataArray: Acquired da for the current tile stream
        """
        # Update the progress bar
        self._future.set_progress(end=self.estimateTime(self._number_of_tiles - i) + time.time())
        # Acquire data array for passed stream
        self._future.running_subf = acqmng.acquire([stream], self._settings_obs)
        das, e = self._future.running_subf.result()  # blocks until all the acquisitions are finished
        if e:
            logging.warning(f"Acquisition for tile {ix}x{iy}, stream {stream.name} partially failed: {e}")

        if self._future._task_state == CANCELLED:
            raise CancelledError()
        try:
            return das[0]  # return first da
        except IndexError:
            raise IndexError(f"Failure in acquiring tile {ix}x{iy}, stream {stream.name}.")

    def _acquireStreamsTile(self, i, ix, iy, streams):
        """
        Calls acquire function and blocks until the data is returned.
        :return: list(DataArray) acquired das for the current tile streams
        """
        # Update the progress bar
        self._future.set_progress(end=self.estimateTime(self._number_of_tiles - i) + time.time())
        # Acquire data array for passed stream
        self._future.running_subf = acqmng.acquire(streams, self._settings_obs)
        das, e = self._future.running_subf.result()  # blocks until all the acquisitions are finished
        if e:
            logging.warning(f"Acquisition for tile {ix}x{iy}, streams partially failed: {e}")

        if self._future._task_state == CANCELLED:
            raise CancelledError()

        return das  # return all das for the tile

    def _getTileDAs(self, i, ix, iy):
        """
        Iterate over each tile stream and construct their data arrays list
        :return: list(DataArray) list of each stream DataArray
        """
        das = []
        das = self._acquireStreamsTile(i, ix, iy, self._streams)
        return das

    def _sortDAs(self, das, ss):
        """
        Sorts das based on priority for stitching, i.e. largest SEM da first, then
        other SEM das, and finally das from other streams.
        das: list of DataArrays
        ss: streams from which the das were extracted

        returns: list of DataArrays, reordered input
        """
        # Add the ACQ_TYPE metadata (in case it's not there)
        # In practice, we check the stream the DA came from, and based on the stream
        # type, fill the metadata
        # TODO: make sure acquisition type is added to data arrays before, so this
        # code can be deleted
        for da in das:
            if model.MD_ACQ_TYPE in da.metadata:
                continue
            for s in ss:
                for sda in s.raw:
                    if da is sda:  # Found it!
                        if isinstance(s, EMStream):
                            da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_EM
                        elif isinstance(s, ARStream):
                            da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_AR
                        elif isinstance(s, SpectrumStream):
                            da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_SPECTRUM
                        elif isinstance(s, FluoStream):
                            da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_FLUO
                        elif isinstance(s, MultipleDetectorStream):
                            if model.MD_OUT_WL in da.metadata:
                                da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_CL
                            else:
                                da.metadata[model.MD_ACQ_TYPE] = model.MD_AT_EM
                        else:
                            logging.warning("Unknown acq stream type for %s", s)
                        break
                if model.MD_ACQ_TYPE in da.metadata:
                    # if da is found, no need to search other streams
                    break
            else:
                logging.warning("Couldn't find the stream for DA of shape %s", da.shape)

        # # Remove the DAs we don't want to (cannot) stitch
        das = [da for da in das if da.metadata[model.MD_ACQ_TYPE] \
               not in (model.MD_AT_AR, model.MD_AT_SPECTRUM)]

        def leader_quality(da):
            """
            return int: The bigger the more leadership
            """
            # For now, we prefer a lot the EM images, because they are usually the
            # one with the smallest FoV and the most contrast
            if da.metadata[model.MD_ACQ_TYPE] == model.MD_AT_EM:
                return numpy.prod(da.shape)  # More pixel to find the overlap
            elif da.metadata[model.MD_ACQ_TYPE]:
                # A lot less likely
                return numpy.prod(da.shape) / 100

        das.sort(key=leader_quality, reverse=True)
        das = tuple(das)
        return das

    def _save_tiles(self, ix, iy, das, stream_cube_id=None):
        """
        Save the acquired data array to disk (for debugging)
        """

        def save_tile(ix, iy, das, stream_cube_id=None):
            if stream_cube_id is not None:
                # Indicate it's a stream cube in the file name
                fn_tile = "%s-cube%d-%.5dx%.5d%s" % (self._fn_bs, stream_cube_id, ix, iy, self._fn_ext)
            else:
                fn_tile = "%s-%.5dx%.5d%s" % (self._fn_bs, ix, iy, self._fn_ext)
            logging.debug("Will save data of tile %dx%d to %s", ix, iy, fn_tile)
            self._exporter.export(os.path.join(self._log_dir, fn_tile), das)

        # Run in a separate thread via the executor
        self._save_executor.submit(save_tile, ix, iy, das, stream_cube_id)

    def _estimateStreamPixels(self, s):
        """
        return (int): the number of pixels the stream will generate during an
          acquisition
        """
        px = 0
        if isinstance(s, MultipleDetectorStream):
            for st in s.streams:
                # For the EMStream of a SPARC MDStream, it's just one pixel per
                # repetition (excepted in case  of fuzzing, but let's be optimistic)
                if isinstance(st, (EMStream, CLStream)):
                    px += 1
                else:
                    px += self._estimateStreamPixels(st)

            if hasattr(s, 'repetition'):
                px *= s.repetition.value[0] * s.repetition.value[1]

            return px
        elif isinstance(s, (ARStream, SpectrumStream)):
            # Temporarily reports 0 px, as we don't stitch these streams for now
            return 0

        if hasattr(s, 'emtResolution'):
            px = numpy.prod(s.emtResolution.value)
        elif hasattr(s, 'detResolution'):
            px = numpy.prod(s.detResolution.value)
        elif model.hasVA(s.detector, "resolution"):
            px = numpy.prod(s.detector.resolution.value)
        elif model.hasVA(s.emitter, "resolution"):
            px = numpy.prod(s.emitter.resolution.value)
        else:
            # This shouldn't happen, but let's "optimistic" by assuming it'll
            # only acquire one pixel.
            logging.info("Resolution of stream %s cannot be determined.", s)
            px = 1

        return px

    def estimateMemory(self):
        """
        Makes an estimate for the amount of memory that will be consumed during
        stitching and compares it to the available memory on the computer.
        :returns (bool) True if sufficient memory available, (float) estimated memory
        """
        mem_sufficient = True
        mem_est = 0.0
        # Number of pixels for acquisition
        pxs = sum(self._estimateStreamPixels(s) for s in self._streams)
        pxs *= self._number_of_tiles

        # Memory calculation
        mem_est = pxs * self.MEMPP
        mem_computer = psutil.virtual_memory().total
        logging.debug("Estimating %g GB needed, while %g GB available",
                      mem_est / 1024 ** 3, mem_computer / 1024 ** 3)
        # Assume computer is using 2 GB RAM for odemis and other programs
        mem_sufficient = mem_est < mem_computer - (2 * 1024 ** 3)

        return mem_sufficient, mem_est

    def estimateTime(self, remaining=None):
        """
        Estimates duration for acquisition and stitching.
        :param remaining: (int > 0) The number of remaining tiles
        :returns: (float) estimated required time
        """
        if remaining is None:
            remaining = self._number_of_tiles

        # After the acquisition of first tile, update the time taken for subsequent tiles based on time taken for
        # previous tiles
        if self.average_acquisition_time:
            return self.average_acquisition_time * remaining

        acq_time = acqmng.estimateTime(self._streams)
        acq_time = acq_time * remaining

        # Estimate stitching time based on number of pixels in the overlapping part
        stitch_time = 0
        max_pxs = 0
        for s in self._streams:
            for sda in s.raw:
                pxs = sda.shape[0] * sda.shape[1]
                if pxs > max_pxs:
                    max_pxs = pxs
        stitch_time = (self._number_of_tiles * max_pxs * self._overlap) / STITCH_SPEED

        try:
            # move_speed is a default speed but not an actual stage speed due to which
            # extra time is added based on observed time taken to move stage from one tile position to another
            move_time = max(self._sfov) * (remaining - 1) / self._move_speed + 0.3 * remaining
            # current tile is part of remaining, so no need to move there
        except ValueError:  # no current streams
            move_time = 0.5

        logging.info(f"The computed time in seconds for tiled acquisition for {remaining} tiles for move is {move_time},"
                     f" acquisition is {acq_time}")

        return acq_time + move_time + stitch_time

    def acquire_all(self):
        """Template method: the main acquisition loop."""
        da_list = []
        positions = self.get_ordered_positions()
        prev_pos = None
        self._save_time = {"acq": [], "stitch": [], "move": [], "save": []}
        # The increase in the number of scanning indices increase with overlap between tiles. The time take
        # by stage to move to different indices also includes the time taken to move when scanning indices increase due
        # to increase in overlap. This means stitching time is included when move time between tiles is observed.
        # Hence, observed time due to stitching is set to zero
        self._save_time["stitch"] = [0]
        move_to_tile_start = None
        start_time = time.time()

        for i, pos in enumerate(positions):
            if i > 0:
                self.average_acquisition_time = (time.time() - start_time) / i

            self.move_to_position(pos, prev_pos, i)
            if move_to_tile_start:
                self._save_time["move"].append(time.time() - move_to_tile_start)
            acquisition_start = time.time()
            self.pre_acquire(pos, i)
            das = self.acquire_at_position(pos, i)
            self.post_acquire(das, pos, i)
            self._save_time["acq"].append(time.time() - acquisition_start)
            save_tile_start = time.time()
            if self._log_path:
                self._save_tiles(pos[0], pos[1], das)
            self._save_time["save"].append(time.time() - save_tile_start)
            da_list.append(self._sortDAs(das, self._streams))

            prev_pos = pos
            move_to_tile_start = time.time()

        return da_list

    def get_ordered_positions(self):
        return self._sort_tile_indices_zigzag(self._tile_indices)

    def move_to_position(self, position, prev_position, index):
        self._moveToTile(position, prev_position or START_INDEX, self._sfov)

    def acquire_at_position(self, position, index):
        ix, iy = position
        return self._getTileDAs(index, ix, iy)

    def pre_acquire(self, position, index):
        return

    def post_acquire(self, das, position, index):
        if index == 0:
            self._sfov = self._updateFov(das, self._sfov)


class FocuserGridTiledAcquisitionController(GridTiledAcquisitionController):
    def __init__(self, streams, region, stage, overlap, sfov, zlevels, focusing_method, focus_points, focus_range, future=None, settings_obs=None, log_path=None, centered_acq=True):
        super().__init__(streams, region, stage, overlap, sfov, future, settings_obs, log_path, centered_acq)
        self._focus_points = focus_points
        self._focus_range = focus_range
        # To check and adjust the focus in between tiles
        if not isinstance(focusing_method, FocusingMethod):
            raise ValueError(f"focusing_method should be of type FocusingMethod, but got {focusing_method}")
        self._focusing_method = focusing_method
        self._focus_stream = next((sd for sd in self._streams if sd.focuser is not None), None)
        self._focus_plane = {}

        if self._focus_stream:
            # save initial focus value to be used in the AutoFocus function
            self._good_focus = self._focus_stream.focuser.position.value['z']
            focuser_range = self._focus_stream.focuser.axes['z'].range
            # Calculate the focus range by half the focus margin on each side of good focus
            focus_rng = (self._good_focus - FOCUS_RANGE_MARGIN / 2, self._good_focus + FOCUS_RANGE_MARGIN / 2)
            # Clip values with focuser_range
            self._focus_rng = (max(focus_rng[0], focuser_range[0]), min((focus_rng[1], focuser_range[1])))
            logging.debug("Calculated focus range ={}".format(self._focus_rng))

            # used in re-focusing method
            self._focus_points = numpy.array(focus_points) if focus_points else None
            # triangulate focus points
            self._tri_focus_points = None
            if focus_points is not None:
                if len(focus_points) >= 3:
                    self._tri_focus_points = Delaunay(self._focus_points[:, :2])
                elif len(focus_points) == 1:
                    # Triangulation needs minimum three points to define a plane
                    # When the number of focus points is less than three
                    # The focus is set constant and based on a single focus point
                    pass
                else:
                    raise ValueError(f"focus_points length {len(focus_points)} is not supported")

        if focusing_method == FocusingMethod.MAX_INTENSITY_PROJECTION and not zlevels:
            raise ValueError("MAX_INTENSITY_PROJECTION requires zlevels, but none passed")
            # Note: we even allow if only one zlevels. It would not do MIP, but
            # that allows for flexibility where the user explicitly wants to disable
            # MIP by setting only one zlevel. Same if there is no focuser.

        if zlevels:
            if self._focus_stream is None:
                logging.warning("No focuser found in any of the streams, only one acquisition will be performed.")
            self._zlevels = numpy.asarray(zlevels)
            self._init_zlevels = numpy.asarray(zlevels)  # zlevels can be relative, therefore keep track of the initial zlevels
        else:
            self._zlevels = numpy.empty(0)
            self._init_zlevels = numpy.empty(0)

        if len(self._zlevels) > 1 and focusing_method != FocusingMethod.MAX_INTENSITY_PROJECTION:
            raise NotImplementedError(
                "Multiple zlevels currently only works with focusing method MAX_INTENSITY_PROJECTION")

        # For "ON_LOW_FOCUS_LEVEL" method: a focus level which is corresponding to a in-focus image.
        self._good_focus_level = None  # float

    def _adjustFocus(self, das, i, ix, iy):
        """
        das (list of DataArray): the data of each stream which has just been acquired
        i (int): the acquisition number
        ix (int): the tile number in x
        iy (int): the tile number in y
        return (list of DataArray): the data of each stream, possibly replaced
          by a new version at a better focus level.
        """
        refocus = False
        # If autofocus explicitly disabled, or MIP => don't do anything
        if self._focusing_method in (FocusingMethod.NONE, FocusingMethod.MAX_INTENSITY_PROJECTION):
            return das
        elif self._focusing_method == FocusingMethod.ON_LOW_FOCUS_LEVEL:
            if i % SKIP_TILES != 0:
                logging.debug("Skipping focus adjustment..")
                return das

            try:
                current_focus_level = MeasureOpticalFocus(das[self._streams.index(self._focus_stream)])
            except IndexError:
                logging.warning("Failed to get image to measure focus on.")
                return das

            if i == 0:
                # Use initial optical focus level to be compared to next tiles
                # TODO: instead of using the first image, use the best 10% images (excluding outliers)
                self._good_focus_level = current_focus_level
                return das

            logging.debug("Current focus level: %s (good = %s)", current_focus_level, self._good_focus_level)

            # Run autofocus if current focus got worse than permitted deviation,
            # or it was very bad (0) originally.
            if (self._good_focus_level != 0 and
                    (self._good_focus_level - current_focus_level) / self._good_focus_level < FOCUS_FIDELITY
            ):
                return das
        elif self._focusing_method == FocusingMethod.ALWAYS:
            pass
        else:
            raise ValueError(f"Unexpected focusing method {self._focusing_method}")

        try:
            self._future.running_subf = AutoFocus(self._focus_stream.detector,
                                                  self._focus_stream.emitter,
                                                  self._focus_stream.focuser,
                                                  good_focus=self._good_focus,
                                                  rng_focus=self._focus_rng,
                                                  method=MTD_EXHAUSTIVE)
            _, focus_pos, _ = self._future.running_subf.result()  # blocks until autofocus is finished

            # Corner case where it started very badly: update the "good focus"
            # as it's likely going to be better.
            if self._good_focus_level == 0:
                self._good_focus_level = focus_pos

            if self._future._task_state == CANCELLED:
                raise CancelledError()
        except CancelledError:
            raise
        except Exception:
            logging.exception("Running autofocus failed on image i= %s." % i)
        else:
            # Reacquire the out of focus tile (which should be corrected now)
            logging.debug("Reacquiring tile %dx%d at better focus %f", ix, iy, focus_pos)
            das = self._getTileDAs(i, ix, iy)

        return das

    def _get_z_on_focus_plane(self, x, y):
        if not self._focus_plane:
            gamma, normal = linalg.fit_plane_lstsq(self._focus_points)
            self._focus_plane["gamma"] = gamma
            self._focus_plane["normal"] = normal
        point_on_plane = (0, 0, self._focus_plane["gamma"])  # where the plane intersects with the z-axis
        z = linalg.get_z_pos_on_plane(x, y, point_on_plane, self._focus_plane["normal"])
        return z

    def _get_triangulated_focus_point(self, x, y):
        """Triangulate focus points and get the z position of the xy-point in the corresponding focus-triangle."""

        point_in_triangle = False
        # Check in 2D if the point is in one of the triangles from the focus points
        for simplex in self._tri_focus_points.simplices:
            triangle = [self._focus_points[:, :2][simplex[0]],
                        self._focus_points[:, :2][simplex[1]],
                        self._focus_points[:, :2][simplex[2]]]
            point_in_triangle = point_in_polygon([x, y], polygon=triangle)
            if point_in_triangle:
                break

        if point_in_triangle is False:
            # TODO determine a better way for dealing with points outside of the focused area, for instance
            #  by using linear extrapolation.
            logging.debug("Acquiring tile outside of focused area, will use plane fitting to find the focus.")
            # If the point is not in one of the triangles fit a plane through all
            # focus points and base the z position on that plane fit.
            z = self._get_z_on_focus_plane(x, y)
        else:
            tr = [self._focus_points[simplex[0]],
                  self._focus_points[simplex[1]],
                  self._focus_points[simplex[2]]]

            z = linalg.get_point_on_plane(x, y, tr)

        return z

    def _refocus(self):
        """Update the z-levels to fit the found focus positions"""
        current_pos = self._stage.position.value
        if self._tri_focus_points:
            z = self._get_triangulated_focus_point(current_pos["x"], current_pos["y"])
        else:
            z = self._focus_points[0][2]
        logging.info(f"Found z focus: {z}")
        self._zlevels = self._get_zstack_levels(z)

    def _get_zstack_levels(self, focus_value):
        """
        Calculate the zstack levels from the current focus position and zsteps value
        :returns (list(float) or None) zstack levels for zstack acquisition.
          return None if only one zstep is requested.
        """
        # When initial z level is one or None, use the current focus value
        if len(self._init_zlevels) <= 1:
            return [focus_value, ]

        zlevels = self._init_zlevels + focus_value

        # Check which focus range is available
        if self._focus_range is not None:
            comp_range = self._focus_range
        else:
            comp_range = self._focus_stream.focuser.axes['z'].range

        # The number of zlevels will remain the same, but the range will be adjusted
        zmin = min(zlevels)
        zmax = max(zlevels)
        if (zmax - zmin) > (comp_range[1] - comp_range[0]):
            # Corner case: it'd be larger than the entire range => limit to the entire range
            zmin = comp_range[0]
            zmax = comp_range[1]
        if zmax > comp_range[1]:
            # Too high => shift down
            shift = zmax - comp_range[1]
            zmin -= shift
            zmax -= shift
        if zmin < comp_range[0]:
            # Too low => shift up
            shift = comp_range[0] - zmin
            zmin += shift
            zmax += shift

        # Create focus zlevels from the given zsteps number
        zlevels = numpy.linspace(zmin, zmax, len(zlevels)).tolist()

        return zlevels

    def _acquireStreamCompressedZStack(self, i, ix, iy, stream):
        """
        Acquire a compressed zstack image for the given stream.
        The method does the following:
            - Move focus over the list of zlevels
            - For each focus level acquire image of the stream
            - Construct xyz cube for the acquired zstack
            - Compress the cube into a single image using 'maximum intensity projection'
        :return DataArray: Acquired da for the current tile stream
        """
        zstack = []
        for z in self._zlevels:
            logging.debug(f"Moving focus for tile {ix}x{iy} to {z}.")
            stream.focuser.moveAbsSync({'z': z})
            da = self._acquireStreamTile(i, ix, iy, stream)
            zstack.append(da)

        if self._future._task_state == CANCELLED:
            raise CancelledError()
        logging.debug(
            f"Zstack acquisition for tile {ix}x{iy}, stream {stream.name} finished, compressing data into a single image.")
        # Convert zstack into a cube
        fm_cube = assembleZCube(zstack, self._zlevels)
        # Save the cube on disk if a log path exists
        if self._log_path:
            self._save_tiles(ix, iy, fm_cube, stream_cube_id=self._streams.index(stream))

        if self._focusing_method == FocusingMethod.MAX_INTENSITY_PROJECTION:
            # Compress the cube into a single image along z-axis (using maximum intensity projection)
            mip_image = img.max_intensity_projection(fm_cube, axis=0)
            if self._future._task_state == CANCELLED:
                raise CancelledError()
            logging.debug(f"Zstack compression for tile {ix}x{iy}, stream {stream.name} finished.")
            return DataArray(mip_image, copy.copy(zstack[0].metadata))
        else:
            # TODO: support stitched Z-stacks
            # For now, the init will raise NotImplementedError in such case
            logging.warning("Zstack returned as-is, while it is not supported")
            return fm_cube

    def _getTileDAs(self, i, ix, iy):
        """
        Iterate over each tile stream and construct their data arrays list
        :return: list(DataArray) list of each stream DataArray
        """
        das = []

        for stream in self._streams:
            if stream.focuser is not None and len(self._zlevels) > 1:
                # Acquire zstack images based on the given zlevels, and compress them into a single da
                da = self._acquireStreamCompressedZStack(i, ix, iy, stream)
            elif stream.focuser is not None and len(self._zlevels) == 1:
                z = self._zlevels[0]
                logging.debug(f"Moving focus for tile {ix}x{iy} to {z}.")
                stream.focuser.moveAbsSync({'z': z})
                da = self._acquireStreamTile(i, ix, iy, stream)
            else:
                # Acquire a single image of the stream
                da = self._acquireStreamTile(i, ix, iy, stream)
            das.append(da)

        return das

    def estimateTime(self, remaining=None):
        """
        Estimates duration for acquisition and stitching.
        :param remaining: (int > 0) The number of remaining tiles
        :returns: (float) estimated required time
        """
        if remaining is None:
            remaining = self._number_of_tiles

        # After the acquisition of first tile, update the time taken for subsequent tiles based on time taken for
        # previous tiles
        if self.average_acquisition_time:
            return self.average_acquisition_time * remaining

        zlevels_dict = {s: self._zlevels for s in self._streams
                        if isinstance(s, (FluoStream))}
        acq_time = acqmng.estimateZStackAcquisitionTime(self._streams, zlevels_dict)
        acq_time = acq_time * remaining

        # Estimate stitching time based on number of pixels in the overlapping part
        stitch_time = 0
        max_pxs = 0
        for s in self._streams:
            for sda in s.raw:
                pxs = sda.shape[0] * sda.shape[1]
                if pxs > max_pxs:
                    max_pxs = pxs
        stitch_time = (self._number_of_tiles * max_pxs * self._overlap) / STITCH_SPEED

        try:
            # move_speed is a default speed but not an actual stage speed due to which
            # extra time is added based on observed time taken to move stage from one tile position to another
            move_time = max(self._sfov) * (remaining - 1) / self._move_speed + 0.3 * remaining
            # current tile is part of remaining, so no need to move there
        except ValueError:  # no current streams
            move_time = 0.5

        logging.info(f"The computed time in seconds for tiled acquisition for {remaining} tiles for move is {move_time},"
                     f" acquisition is {acq_time}")

        return acq_time + move_time + stitch_time

    def pre_acquire(self, position, index):
        if self._focus_points is not None:
            self._refocus()

    def post_acquire(self, das, position, index):
        if index == 0:
            self._sfov = self._updateFov(das, self._sfov)
        if self._focus_stream:
            das = self._adjustFocus(das, index, *position)

class SparcGridTiledAcquisitionController(GridTiledAcquisitionController):
    """
    Grid tiled acquisition controller for SPARC microscope, which includes a specific acquisition order of tiles and a specific
    way to adjust focus in between tiles.
    """
    def __init__(self, streams, region, stage, overlap, sfov, z_correction=None, future=None, settings_obs=None, log_path=None):
        super().__init__(streams, region, stage, overlap, sfov, future, settings_obs, log_path)
        self._z_correction = z_correction
        # SPARC specific acquisition order of tiles, which is different from the default zigzag order

class SparcTileMapAcquisitionController(BaseTileAcquisitionController):
    """
    Tile map acquisition controller for SPARC microscope, which includes a specific acquisition order of tiles and a specific
    way to adjust focus in between tiles.
    """
    def __init__(self, streams, region, sfov, main_data, ccd_stream, overlap, survey_points_dist, filename, z_correction=None, settings_obs=None, log_path=None, future=None):
        super().__init__(streams, main_data.stage)
        self._sfov = sfov
        self._overlap = overlap
        self._z_correction = z_correction
        self._main_data = main_data
        self._ccd_stream = ccd_stream
        self._region = region
        self._survey_points_dist = survey_points_dist
        self._log_dir = os.path.dirname(filename)
        self._fn_bs, self._fn_ext = udataio.splitext(filename)

        self._executor = CancellableThreadPoolExecutor(max_workers=1)  # one task at a time
        self._save_executor = ThreadPoolExecutor(max_workers=5)

    @staticmethod
    def _make_cancel_move(future: model.ProgressiveFuture) -> Callable:
        """
        Return a task_canceller function that cancels sub-futures stored on
        *future* by the movement helper.
        """
        def cancel_move(f):
            for attr in ("running_subf_z", "running_subf_xy", "running_subf_target_z"):
                subf = getattr(f, attr, None)
                if subf is not None:
                    logging.debug("Cancelling %s on future %s.", attr, f)
                    subf.cancel()
        return cancel_move

    def _do_safe_move(
        self,
        future: model.ProgressiveFuture,
        xy_pos: Dict,
        target_z: float = None,
    ) -> None:
        """
        Perform a collision-safe XY (and optionally Z) stage movement:

        1. Move up by SAFE_Z_OFFSET (prevents dragging across sample surface).
        2. Move to *xy_pos*.
        3. If *target_z* is given, move to that absolute Z position.

        Sub-futures are stored on *future* so they can be cancelled via
        _make_cancel_move().

        :param future: ProgressiveFuture that owns this movement.
        :param xy_pos: Dict with at least ``"x"`` and ``"y"`` keys (metres).
        :param target_z: Optional absolute Z target in metres.
        """
        stage = self._stage

        future.running_subf_z = stage.moveRel({"z": self.SAFE_Z_OFFSET})
        future.running_subf_z.result()

        future.running_subf_xy = stage.moveAbs(xy_pos)
        future.running_subf_xy.result()

        if target_z is not None:
            future.running_subf_target_z = stage.moveAbs({"z": target_z})
            future.running_subf_target_z.result()

    def _do_move(
        self,
        future: model.ProgressiveFuture,
        xy_pos: Dict,
        predecessor: model.ProgressiveFuture = None,
    ) -> None:
        """
        Perform a direct XY move.
        Sub-future is stored on *future* so it can be cancelled via _make_cancel_move().
        :param future: ProgressiveFuture that owns this movement.
        :param xy_pos: Dict with at least ``"x"`` and ``"y"`` keys (metres).
        """
        if predecessor is not None:
            predecessor.result()  # propagates any exception / cancellation

        future.running_subf_xy = self._stage.moveAbs(xy_pos)
        future.running_subf_xy.result()

    def _build_z_align_axis(self) -> "AlignmentAxis":
        """
        Compute Z bounds from the current working-distance delta and return
        an AlignmentAxis ready for use with ParabolicMirrorAlignmentTask.

        :return: AlignmentAxis configured for the current stage Z position.
        :raises RuntimeError: if mirror calibration metadata is missing.
        """
        main_data = self._main_data
        mirror_md = main_data.mirror.getMetadata()
        calib = mirror_md[model.MD_CALIB]
        min_step_size = calib["auto_align_min_step_size"]
        ebeam_wd_calib = calib["ebeam_working_distance"]

        current_ebeam_wd = main_data.ebeam_focus.position.value["z"]
        current_stage_z = main_data.stage.position.value["z"]
        wd_delta = ebeam_wd_calib - current_ebeam_wd
        good_z = current_stage_z + wd_delta

        focus_dist = main_data.lens.focusDistance.value if main_data.lens else 500e-6
        z_min = good_z - focus_dist * 0.3
        z_max = good_z + focus_dist

        return AlignmentAxis("z", min_step_size["z"], main_data.stage,
                             abs_bounds=(z_min, z_max))

    def _do_z_align(
        self,
        predecessor: model.ProgressiveFuture,
        z_align_task: "ParabolicMirrorAlignmentTask",
        search_range: float = 100e-6,
        max_iter: int = 50,
    ) -> None:
        """
        Wait for *predecessor* to finish, switch to mirror-align optical path,
        activate the CCD stream and run the Z-alignment task.

        The CCD stream is always deactivated in the finally block.

        :param predecessor: Future that must complete before alignment starts.
        :param z_align_task: Configured ParabolicMirrorAlignmentTask.
        :param search_range: Z search range passed to align_mirror (metres).
        :param max_iter: Maximum alignment iterations.
        """
        predecessor.result()  # propagates any exception / cancellation

        self._main_data.opm.setPath("mirror-align").result()
        time.sleep(5)

        self._ccd_stream.is_active.value = True
        self._ccd_stream.should_update.value = True
        try:
            z_align_task.align_mirror(search_range=search_range, max_iter=max_iter)
        except StopIteration:
            logging.debug("StopIteration raised during Z-alignment (normal early-stop).")
        finally:
            self._ccd_stream.is_active.value = False
            self._ccd_stream.should_update.value = False

    def _perform_z_survey_point(
        self,
        survey_pos: Dict,
    ) -> model.ProgressiveFuture:
        """
        Perform Z-height measurement at a survey point with safe movements.

        Moves safely to the survey point with configured safe Z offset,
        then acquires the Z-height through mirror alignment.

        :param survey_pos: Dict with 'x', 'y' keys for the XY position
        :return: model.ProgressiveFuture representing the survey operation
        """
        z_align_axis = self._build_z_align_axis()

        # --- movement future ---
        f_move = model.ProgressiveFuture()
        f_move.task_canceller = self._make_cancel_move(f_move)

        # --- z-alignment future ---
        f_z_align = model.ProgressiveFuture()
        f_z_align.n_steps = 0
        f_z_align.current_step = 0
        z_align_task = ParabolicMirrorAlignmentTask(
            [z_align_axis],
            self._main_data.ccd,
            f_z_align,
            stop_early=True,
            save_images=True,
        )
        f_z_align.task_canceller = z_align_task.cancel

        self._executor.submitf(f_move, self._do_safe_move, f_move, survey_pos)
        self._executor.submitf(f_z_align, self._do_z_align, f_move, z_align_task)

        return model.ProgressiveBatchFuture({f_move: 25, f_z_align: 60})

    def _get_z_alignment_position(self, tile_center: Tuple[float, float],
                                   fov: Tuple[float, float],
                                   corner: str = "top-left") -> Tuple[float, float]:
        """
        Calculate a safe Z-alignment position offset from the tile center based on the FoV and corner choice.

        This function computes an offset position for Z-alignment that is safely away from the tile,
        taking into account the FoV size and the desired corner for alignment.

        :param tile_center: (x, y) coordinates of the tile center in meters
        :param fov: (fov_x, fov_y) size of the field of view in meters
        :param corner: Desired corner for Z-alignment ("top-left", "top-right", "bottom-left", "bottom-right")
        :return: (x, y) coordinates for Z-alignment position in meters
        """
        # Calculate FoV offsets
        offset_x = fov[0] * 1.5
        offset_y = fov[1] * 1.5

        # Map corner names to offsets
        corner_offsets = {
            "top-left": (-offset_x, offset_y),
            "top-right": (offset_x, offset_y),
            "bottom-left": (-offset_x, -offset_y),
            "bottom-right": (offset_x, -offset_y),
        }

        if corner not in corner_offsets:
            logging.warning(f"Unknown corner '{corner}', defaulting to 'top-left'")
            corner = "top-left"

        offset = corner_offsets[corner]
        z_align_pos = (tile_center[0] + offset[0], tile_center[1] + offset[1])

        return z_align_pos

    def _do_acquire(
        self,
        predecessor: model.ProgressiveFuture,
        tile_acq_task: AcquisitionTask,
        tile_path: str,
        da_list: list,
    ) -> None:
        """
        Wait for the preceding movement/alignment future to complete, then
        acquire one tile and schedule an asynchronous save.

        Called by the executor as a chained sub-future.  Any exception or
        cancellation raised by *f_pre_acq* is propagated automatically via
        ``f_pre_acq.result()``.

        Acquired DataArrays are appended to *da_list* so the caller can
        later stitch all tiles together.  The file is written in the
        background (daemon thread) so acquisition of the next tile can
        begin immediately.

        :param predecessor: Future that must complete before alignment starts.
        :param tile_acq_task: Configured AcquisitionTask for this tile.
        :param tile_path: Destination file path for this tile.
        :param da_list: Shared list that accumulates all acquired DataArrays.
        :raises AcquisitionError: If the underlying acquisition task fails
            with a non-recoverable error.
        """
        def save_tile(tile_path, das):
            """Save tile data to disk"""
            exporter = dataio.find_fittest_converter(tile_path)
            logging.debug("Will save data of tile %s", tile_path)
            exporter.export(tile_path, das)

        predecessor.result()  # propagates any exception / cancellation
        das, e = tile_acq_task.run()
        if e:
            logging.warning(f"Acquisition for tile {tile_path} partially failed: {e}")
        if not das:
            raise ValueError(f"Acquisition for tile {tile_path} failed with no data: {e}")
        da_list.extend(das)
        if self._save_executor is not None:
            self._save_executor.submit(save_tile, tile_path, das)

    def _acquire_single_tile(
        self,
        tile_pos: Dict,
        streams,
        tile_path,
        da_list,
        fov: Tuple[float, float] = None,
        target_z: float = None,
        enable_z_alignment: bool = False,
    ) -> model.ProgressiveFuture:
        """
        Unified tile acquisition function that handles both calibration-based Z-alignment
        and direct Z-positioning modes.

        This replaces both _custom_tiled_target_z_acquisition and _custom_tiled_acquisition,
        providing a clean, modular approach to tile acquisition with or without Z-alignment.

        When Z-alignment is enabled, the function moves to a safe corner position (offset
        from the tile center by FoV/2) to perform Z-height alignment. This prevents damage
        to the sample in the actual imaging region.

        :param tile_pos: Dict with 'x', 'y' keys for the tile XY position (center)
        :param streams: List of streams to acquire
        :param tile_path: File path to save the tile data
        :param da_list: List to accumulate acquired data arrays
        :param fov: Tuple of (width, height) field of view in meters.
                   Required when enable_z_alignment=True. Used to calculate safe corner position.
        :param target_z: Target Z position (used when enable_z_alignment=False)
        :param enable_z_alignment: If True, use mirror alignment for Z at a safe corner position;
                                   if False, move to target_z directly at tile center
        :return: model.ProgressiveFuture tracking the acquisition
        """
        main_data = self._main_data
        futures: Dict[model.ProgressiveFuture, int] = {}

        # --- movement future (always present) ---
        f_move = model.ProgressiveFuture()
        f_move.task_canceller = self._make_cancel_move(f_move)
        futures[f_move] = 25

        # --- acquisition future (always present) ---
        f_acq = model.ProgressiveFuture()
        acq_task = acqmng.AcquisitionTask(streams, f_acq, main_data.settings_obs)
        f_acq.task_canceller = acq_task.cancel
        futures[f_acq] = sum(acq_task._streamTimes.values())

        if enable_z_alignment:
            # Move to a safe corner position offset from the tile centre
            z_align_pos_xy = self._get_z_alignment_position(
                (tile_pos["x"], tile_pos["y"]), fov, corner="top-left"
            )
            move_to = {"x": z_align_pos_xy[0], "y": z_align_pos_xy[1]}

            # --- z-alignment future ---
            z_align_axis = self._build_z_align_axis()
            f_z_align = model.ProgressiveFuture()
            f_z_align.n_steps = 0
            f_z_align.current_step = 0
            z_align_task = ParabolicMirrorAlignmentTask(
                [z_align_axis], main_data.ccd, f_z_align,
                stop_early=True, save_images=True,
            )
            f_z_align.task_canceller = z_align_task.cancel
            futures[f_z_align] = 60

            # --- move to tile center future (after Z-alignment) ---
            f_move_to_tile = model.ProgressiveFuture()
            f_move_to_tile.task_canceller = self._make_cancel_move(f_move_to_tile)
            futures[f_move_to_tile] = 5

            self._executor.submitf(f_move, self._do_safe_move, f_move, move_to)
            self._executor.submitf(f_z_align, self._do_z_align, f_move, z_align_task)
            self._executor.submitf(f_move_to_tile, self._do_move, f_move_to_tile, tile_pos, f_z_align)
            self._executor.submitf(f_acq, self._do_acquire, f_move_to_tile, acq_task, tile_path, da_list)
        else:
            move_to = tile_pos
            self._executor.submitf(f_move, self._do_safe_move, f_move, move_to, target_z)
            self._executor.submitf(f_acq, self._do_acquire, f_move, acq_task, tile_path, da_list)

        return model.ProgressiveBatchFuture(futures)

    def _generate_survey_points(self, dist: float) -> List[Tuple[float, float]]:
        """
        Generates survey points for the Z-Map.
        - If shape is a Point or Line (Area=0): Returns the Centroid.
        - If shape is a Polygon (Area>0): Returns grid points strictly inside the polygon.
        """
        if not self._region:
            return []

        coords = list(self._region.values())

        # Convex Hull wraps the points.
        # It returns a Point, LineString, or Polygon.
        geom = MultiPoint(coords).convex_hull

        # CASE 1: Point or Line (Area is 0)
        # As requested: just return the midpoint.
        if geom.area == 0:
            return [(geom.centroid.x, geom.centroid.y)]

        # CASE 2: Valid Polygon (Area > 0)
        # Generate the grid strictly inside.
        minx, miny, maxx, maxy = geom.bounds

        # Use arange for fixed steps.
        x_arr = numpy.arange(minx, maxx + dist, dist)
        y_arr = numpy.arange(miny, maxy + dist, dist)

        valid_points = []
        for px in x_arr:
            for py in y_arr:
                # .contains is strict (points on the edge are False).
                # Use .intersects or buffer if you want edge points,
                # but usually strict inside is safer for microscope limits.
                if geom.contains(Point(px, py)) or geom.touches(Point(px, py)):
                    valid_points.append((px, py))

        valid_points.append((geom.centroid.x, geom.centroid.y))

        return valid_points

    def get_ordered_positions(self):
        return self._region.items()

    def move_to_position(self, position, prev_position, index):
        return super().move_to_position(position, prev_position, index)

    def acquire_at_position(self, position, index):
        return super().acquire_at_position(position, index)

    def pre_acquire(self):
        known_points = [] # (x, y)
        known_z = []      # z
        survey_points = self._generate_survey_points(dist=self._survey_points_dist)

        # Get current position for intelligent path planning
        current_xy = (self._stage.position.value["x"],
                        self._stage.position.value["y"])

        # Order survey points using nearest-neighbor to minimize travel distance
        ordered_survey_points = order_positions(
            survey_points, current_xy
        )

        logging.info(f"Measuring {len(ordered_survey_points)} survey points in optimized order")

        previous_z = self._stage.position.value["z"]
        # Measure Z at each survey point in smart order
        for i, survey_point in enumerate(ordered_survey_points, start=1):
            ft = self._perform_z_survey_point({"x": survey_point[0], "y": survey_point[1]})
            ft.result()

            measured_z = self._stage.position.value["z"]
            if abs(measured_z - previous_z) > 0.6 * self.SAFE_Z_OFFSET:
                measured_z = previous_z  # Discard outlier and use previous Z
                self._stage.moveAbs({"z": measured_z}).result()  # Move back to previous Z
                logging.debug(f"Discarded Z measurement at point {survey_point}, using previous Z={measured_z:.6e}")
            known_z.append(measured_z)
            known_points.append(survey_point)
            previous_z = measured_z

            logging.debug(f"Survey point {i}: pos={survey_point}, z={measured_z:.6e}")

        # Create interpolator from the surveyed points
        self._z_interpolator = create_z_interpolator(known_points, known_z)

        try:
            x = numpy.array(known_points)[:, 0]
            y = numpy.array(known_points)[:, 1]
            z = numpy.array(known_z)
            plt.figure(figsize=(6, 5))
            plt.tricontourf(x, y, z, levels=14, cmap="viridis")
            plt.colorbar(label="Z height (m)")
            plt.scatter(x, y, c="red", label="Survey Points")
            plt.title("Z-Map")
            plt.xlabel("X (m)")
            plt.ylabel("Y (m)")
            plt.axis("equal")
            plt.legend()
            plt.tight_layout()
            z_map_path = os.path.join(self._log_dir, f"{self._fn_bs}_zmap.png")
            plt.savefig(z_map_path)
            plt.close()
            logging.info(f"Saved Z-map visualization to {z_map_path}")
        except Exception as e:
            logging.warning(f"Failed to create Z-map visualization: {e}")

        self._stage.moveRel({"z": 300e-6}).result()  # Move away from sample before moving to tiles

    def post_acquire(self, das, position, index):
        return super().post_acquire(das, position, index)

    def acquire_all(self):
        self.pre_acquire()
        da_list = []
        for tile_num, tile_pos in self.get_ordered_positions():

            fn_tile = "%s-%d-%.3f-%.3f%s" % (self._fn_bs, tile_num, tile_pos[0] * 1e6,
                                                tile_pos[1] * 1e6, self._fn_ext)
            tile_path = os.path.join(self._log_dir, fn_tile)

            # Interpolate Z height at this tile position
            target_z = float(self._z_interpolator(tile_pos[0], tile_pos[1]))

            logging.debug(f"Tile {tile_num}: pos={tile_pos}, target_z={target_z:.6e}")

            # Acquire tile using unified function (without z-alignment)
            ft = self._acquire_single_tile(
                {"x": tile_pos[0], "y": tile_pos[1]},
                self._streams,
                tile_path,
                da_list,
                fov=self._sfov,
                target_z=target_z,
                enable_z_alignment=False,
            )

            # Wait for this tile to complete before moving to next
            ft.result()
        return da_list

class SparcTiledAcquisitionTask(BaseTiledAcquisitionTask):
    """
    Tiled acquisition task for SPARC microscope, which includes a specific acquisition order of tiles and a specific
    way to adjust focus in between tiles.
    """
    def __init__(self, streams, stage, region, sfov, overlap, z_correction=None, settings_obs=None, log_path=None, future=None,
                 registrar=None, weaver=None):
        super().__init__(registrar, weaver)
        
        
        if isinstance(region, tuple) and len(region) == 4:
            self.ctrl = SparcGridTiledAcquisitionController(streams, region, stage, overlap, sfov, z_correction, future, settings_obs, log_path)
        elif isinstance(region, dict) and all(isinstance(point, tuple) and len(point) == 2 for point in region.values()) and all(isinstance(key, int) for key in region.keys()):
            self.ctrl = SparcTileMapAcquisitionController(streams, region, stage, sfov, overlap, z_correction, settings_obs, log_path, future)
        else:
            raise ValueError(
                "Region must either be a bounding box (xmin, ymin, xmax, ymax) or a list of (x, y) points."
            )

class TiledAcquisitionTask(BaseTiledAcquisitionTask):
    """
    The goal of this task is to acquire a set of tiles then stitch them together
    """

    def __init__(self, streams, stage, region, overlap, settings_obs=None, log_path=None, future=None, zlevels=None,
                 registrar=REGISTER_GLOBAL_SHIFT, weaver=WEAVER_MEAN, focusing_method=FocusingMethod.NONE,
                 focus_points=None, focus_range=None, centered_acq=True):
        """
        :param streams: (list of Streams) the streams to acquire
        :param stage: (Actuator) the sample stage to move to the possible tiles locations
        :param region: Either:
            - (Tuple[float, float, float, float]) Bounding box as (xmin, ymin, xmax, ymax)
            - (List[Tuple[float, float]]) List of (x, y) points defining a polygon
        :param overlap: (float) the amount of overlap between each acquisition
        :param settings_obs: (SettingsObserver or None) class that contains a list of all VAs
            that should be saved as metadata
        :param log_path: (string) directory and filename pattern to save acquired images for debugging
        :param future: (ProgressiveFuture or None) future to track progress, pass None for estimation only
        :param zlevels: (list(float) or None) focus z positions required zstack acquisition.
           Currently, can only be used if focusing_method == MAX_INTENSITY_PROJECTION.
           If focus_points is defined, zlevels is adjusted relative to the focus_points.
        :param registrar: (REGISTER_* or None) type of registration method. If registrar and weaver are None, do not stitch.
        :param weaver: (WEAVER_* or None) type of weaving method. If registrar and weaver are None, do not stitch.
        :param focusing_method: (FocusingMethod) Defines when will the autofocuser be run.
           The autofocuser uses the first stream with a .focuser.
           If MAX_INTENSITY_PROJECTION is used, zlevels must be provided too.
        :param focus_points: (list of tuples) list of focus points corresponding to the known (x, y, z) at good focus.
              If None, the focus will not be adjusted based on the stage position.
        :param focus_range: (tuple) tuple defining the focus range for good focus.
        :param centered_acq: (bool) If True, center the acquisition area on the given region; any extra area is added
            symmetrically to all sides of the bounding box. If False, the top-left of the acquisition area is aligned
            with the top-left of the bounding box.
        """
        super().__init__(registrar, weaver)

        # Get the smallest field of view
        sfov = self.guessSmallestFov(streams)
        logging.debug("Smallest FoV: %s", sfov)

        if focus_points is not None:
            self.ctrl = FocuserGridTiledAcquisitionController(streams, region, stage, overlap, sfov, zlevels, focusing_method,
                                                            focus_points, focus_range, future, settings_obs, log_path, centered_acq)
        else:
            self.ctrl = GridTiledAcquisitionController(streams, region, stage, overlap, sfov, future, settings_obs, log_path, centered_acq)

        # Note: we used to change the stream horizontalFoV VA to the max, if available (eg, SEM).
        # However, it's annoying if the caller actually cares about the FoV (eg,
        # because it wants a large number of pixels, or there are artifacts at
        # the largest FoV). In addition, it's quite easy to do for the caller anyway.
        # Something like this:
        # for stream in self._streams:
        #     if model.hasVA(stream, "horizontalFoV"):
        #         stream.horizontalFoV.value = stream.horizontalFoV.clip(max(self._area_size))

    def _cancelAcquisition(self, future):
        """
        Canceler of acquisition task.
        """
        logging.debug("Canceling acquisition...")

        with future._task_lock:
            if future._task_state == FINISHED:
                return False
            future._task_state = CANCELLED
            future.running_subf.cancel()
            logging.debug("Acquisition cancelled.")
        return True

    def estimateTime(self):
        return self.ctrl.estimateTime()

    def estimateMemory(self):
        return self.ctrl.estimateMemory()

    def run(self):
        """
        Runs the tiled acquisition procedure
        returns:
            (list of DataArrays): a stitched data for each stream acquisition
        raise:
            CancelledError: if acquisition is cancelled
            Exception: if it failed before any result were acquired
        """
        if not self.ctrl._future:
            return
        self.ctrl._future._task_state = RUNNING
        st_data = []
        try:
            # Acquire the needed tiles
            da_list = self.ctrl.acquire_all()

            if self._registrar is not None and self._weaver is not None:
                if not da_list or not da_list[0]:
                    logging.warning("No stream acquired that can be used for stitching.")
                else:
                    logging.info("Acquisition completed, now stitching...")
                    # Stitch the acquired tiles
                    self.ctrl._future.set_progress(end=self.ctrl.estimateTime(0) + time.time())
                    st_data = self._stitchTiles(da_list)

            if self.ctrl._future._task_state == CANCELLED:
                raise CancelledError()
        except CancelledError:
            logging.debug("Acquisition cancelled")
            raise
        except Exception as ex:
            logging.exception("Acquisition failed.")
            self.ctrl._future.running_subf.cancel()
            raise
        finally:
            logging.info("Tiled acquisition ended")
            avg_per_action = {key: statistics.mean(val) for key, val in self.ctrl._save_time.items() if len(val) > 0}
            logging.debug(f"The actual time taken per tile for each action is {self.ctrl._save_time}")
            logging.debug(f"The average time taken per tile for each action is {avg_per_action}")
            logging.debug(f"The average time taken per tile is {self.ctrl.average_acquisition_time}")
            if self.ctrl._save_executor is not None:
                self.ctrl._save_executor.shutdown()
            with self.ctrl._future._task_lock:
                self.ctrl._future._task_state = FINISHED
        return st_data


def estimateTiledAcquisitionTime(*args, **kwargs):
    """
    Estimate the time required to complete a tiled acquisition task
    Parameters are the same as for TiledAcquisitionTask
    :returns: (float) estimated required time
    """
    # Create a tiled acquisition task with future = None
    task = TiledAcquisitionTask(*args, **kwargs)
    return task.estimateTime()


def estimateTiledAcquisitionMemory(*args, **kwargs):
    """
    Estimate the amount of memory required to complete a tiled acquisition task
    Parameters are the same as for TiledAcquisitionTask
    :returns (bool) True if sufficient memory available, (float) estimated memory
    """
    # Create a tiled acquisition task with future = None
    task = TiledAcquisitionTask(*args, **kwargs)
    return task.estimateMemory()


def acquireTiledArea(streams, stage, area, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
                     registrar=REGISTER_GLOBAL_SHIFT, weaver=WEAVER_MEAN, focusing_method=FocusingMethod.NONE,
                     focus_points=None, focus_range=None, centered_acq=True):
    """
    Start a tiled acquisition task for the given streams (SEM or FM) in order to
    build a complete view of the TEM grid. Needed tiles are first acquired for
    each stream, then the complete view is created by stitching the tiles.

    Parameters are the same as for TiledAcquisitionTask
    :return: (ProgressiveFuture) an object that represents the task, allow to
        know how much time before it is over and to cancel it. It also permits
        to receive the result of the task, which is a list of model.DataArray:
        the stitched acquired tiles data
    """
    # Create a progressive future with running sub future
    future = model.ProgressiveFuture()
    # Create a tiled acquisition task
    task = TiledAcquisitionTask(streams, stage, area, overlap, settings_obs, log_path, future=future, zlevels=zlevels,
                                registrar=registrar, weaver=weaver, focusing_method=focusing_method,
                                focus_points=focus_points, focus_range=focus_range, centered_acq=centered_acq)
    future.task_canceller = task._cancelAcquisition  # let the future cancel the task
    # Estimate memory and check if it's sufficient to decide on running the task
    mem_sufficient, mem_est = task.estimateMemory()
    if not mem_sufficient:
        raise IOError("Not enough RAM to safely acquire the overview: %g GB needed" % (mem_est / 1024 ** 3,))

    future.set_progress(end=task.estimateTime() + time.time())
    # connect the future to the task and run in a thread
    executeAsyncTask(future, task.run)

    return future


def estimateOverviewTime(*args, **kwargs):
    """
    Estimate the time required to complete a overview acquisition for a list of areas.
    Parameters are the same as for acquireOverview
    :returns: (float) estimated required time
    """
    # Create an overview acquisition task with future = None
    task = AcquireOverviewTask(*args, **kwargs)
    return task.estimate_time()


def acquireOverview(streams, stage, areas, focus, detector, overlap=0.2, settings_obs=None, log_path=None, zlevels=None,
                    registrar=REGISTER_GLOBAL_SHIFT, weaver=WEAVER_MEAN, focusing_method=FocusingMethod.NONE,
                    use_autofocus: bool = False, focus_points_dist: float = MAX_DISTANCE_FOCUS_POINTS):
    """
    Start autofocus and tiled acquisition tasks for each area in the list of area which is
    given by the input argument areas.

    :param streams: (list of Stream) the streams to acquire
    :param stage: (Actuator) the stage to move
    :param areas: (list of areas) List of areas where an overview image should be acquired.
        Each area is defined by 4 floats :left, top, right, bottom positions
        of the acquisition area (in m)
    :param focus: (Actuator) the focus actuator
    :param detector: (Detector) the detector to use for the acquisition
    :param overlap: (0<float<1) the overlap between tiles (in percentage)
    :param settings_obs: (dict) the settings for the acquisition
    :param log_path: (str) path to the log file
    :param zlevels: (list of floats) the zlevels to use for the stitching, in meters and as relative positions (from the
        focus points
    :param registrar: (str) the type of registration to use
    :param weaver: (str) the type of weaver to use
    :param focusing_method: (str) the focusing method to use
    :param use_autofocus: (bool) whether to use autofocus or not
    :return: (ProgressiveFuture) an object that represents the task, allow to
        know how much time before it is over and to cancel it. It also permits
        to receive the result of the task, which is a list of model.DataArray:
        the stitched acquired tiles data for each area
    """
    # Create a progressive future with running sub future
    future = model.ProgressiveFuture()
    task = AcquireOverviewTask(streams, stage, areas, focus, detector, future, overlap, settings_obs,
                               log_path, zlevels,
                               registrar, weaver, focusing_method, use_autofocus, focus_points_dist)
    future.task_canceller = task.cancel  # let the future cancel the task

    future.set_progress(end=task.estimate_time() + time.time())
    # connect the future to the task and run in a thread
    executeAsyncTask(future, task.run)

    return future


class AcquireOverviewTask(object):
    """
    Create a task to run autofocus and tiled acquisition for each area in the list of areas
    """

    def __init__(self, streams, stage, areas, focus, detector, future=None,
                 overlap=0.2, settings_obs=None, log_path=None,
                 zlevels=None, registrar=REGISTER_GLOBAL_SHIFT, weaver=WEAVER_MEAN,
                 focusing_method=FocusingMethod.NONE, use_autofocus: bool = False,
                 focus_points_dist: float = MAX_DISTANCE_FOCUS_POINTS):
        # site and feature means the same
        # list of time taken for each task i.e autofocus and tiled acquisition for each area
        self.time_per_task = []
        self._stage = stage
        self._future = future
        if future is not None:
            self._future.running_subf = model.InstantaneousFuture()
            self._future._task_lock = threading.Lock()
            self._future._task_state = PENDING
        self.streams = streams
        self.areas = areas  # list of areas
        self._det = detector  # TODO to delete after ccd.data components is used from streams in do_autofocus_roi
        self._focus = focus
        active_rng = self._focus.getMetadata().get(model.MD_POS_ACTIVE_RANGE, None)

        if active_rng and "z" in active_rng:
            self.focus_rng = tuple(active_rng["z"])
        else:
            # No absolute range is available, so use a relative range (e.g. as on METEOR
            rel_rng = self._focus.getMetadata().get(model.MD_SAFE_REL_RANGE, SAFE_REL_RANGE_DEFAULT)
            self.focus_rng = (self._focus.position.value["z"] + rel_rng[0],
                              self._focus.position.value["z"] + rel_rng[1])

        self.conf_level = 0.8
        self.focusing_method = focusing_method
        self._use_autofocus: bool = use_autofocus
        self._focus_points = []  # list of focus points per each area in areas
        self._total_nb_focus_points = 0
        for area in areas:
            focus_points = generate_triangulation_points(focus_points_dist, area)
            self._total_nb_focus_points += len(focus_points)
            self._focus_points.append(focus_points)
        self._overlap = overlap
        self._settings_obs = settings_obs
        self._log_path = log_path
        self._zlevels = zlevels
        self._registrar = registrar
        self._weaver = weaver

    def cancel(self, future):
        """
        Canceler of acquisition task.
        """
        logging.debug("canceling acquisition overview...")

        with future._task_lock:
            if future._task_state == FINISHED:
                return False
            future._task_state = CANCELLED
            future.running_subf.cancel()
            logging.debug("acquisition overview cancelled.")
        return True

    def estimate_time(self) -> float:
        """
        Estimates the time for the rest of the acquisition.
        :return: (float) the estimated time for the rest of the acquisition
        """
        autofocus_time = 0
        self.time_per_task= []
        if self._use_autofocus:
            autofocus_time = estimate_autofocus_in_roi_time(self._total_nb_focus_points, self._det, self._focus, self.focus_rng)
        tiled_time = 0
        for area in self.areas:
            tiled_acq_time = estimateTiledAcquisitionTime(
                                    self.streams, self._stage, area,
                                    focus_range=self.focus_rng,
                                    overlap=self._overlap,
                                    settings_obs=self._settings_obs,
                                    log_path=self._log_path,
                                    zlevels=self._zlevels,
                                    registrar=self._registrar,
                                    weaver=self._weaver,
                                    focusing_method=self.focusing_method)
            tiled_time += tiled_acq_time

            if self._use_autofocus:
                self.time_per_task.extend([autofocus_time, tiled_acq_time])
            else:
                self.time_per_task.append(tiled_acq_time)


        logging.debug(f"Estimated autofocus time: {autofocus_time * len(self.areas)} s, Tiled acquisition time: {tiled_time} s")
        acquisition_time = autofocus_time + tiled_time

        return acquisition_time

    def _pass_future_progress(self, future, start: float, end: float) -> None:
        # update the main future with running sub future time estimate and time estimation of remaining tasks
        self._future.set_progress(end=end + sum(self.time_per_task))

    def run(self):
        """
        The main function of the task class, which will be called by the future asynchronously
        """
        if not self._future:
            raise ValueError("To execute the task, you should pass a Future at init")

        self._future._task_state = RUNNING
        da_rois = []
        try:
            self._future.set_end_time(time.time() + self.estimate_time())
            # create a for loop for roi to create sub futures
            for idx, roi in enumerate(self.areas):

                focus_points = None
                if self._use_autofocus:
                    # cancel the sub future
                    with self._future._task_lock:
                        if self._future._task_state == CANCELLED:
                            raise CancelledError()
                        # is_active set to True will keep the acquisition going on
                        self.streams[0].is_active.value = True
                        logging.debug(f"Autofocus is running for roi number {idx}, with bounding box: {roi} [m]")
                        # run autofocus for the selected roi

                        self._future.running_subf = autofocus_in_roi(roi,
                                                                    self._stage,
                                                                    self._det,
                                                                    self._focus,
                                                                    self.focus_rng,
                                                                    self._focus_points[idx],
                                                                    self.conf_level)
                        # remove the current autofocus time from the list and assign the remaining time to the future
                        # along with the time update of the current autofocus sub future
                        self.time_per_task.pop(0)
                        self._future.running_subf.add_update_callback(self._pass_future_progress)

                    try:
                        focus_points_per_area = len(self._focus_points[idx])
                        max_wait_t = estimate_autofocus_in_roi_time(focus_points_per_area, self._det, self._focus, self.focus_rng) * 3 + 1
                        focus_points = self._future.running_subf.result(max_wait_t)
                    except TimeoutError:
                        logging.debug(f"Autofocus timed out for roi number {idx}, with bounding box: {roi} [m]")
                        raise
                    except Exception:
                        logging.debug(f"Autofocus failed for roi number {idx}, with bounding box: {roi} [m]")
                        raise
                    finally:
                        self.streams[0].is_active.value = False

                # cancel the sub future
                with self._future._task_lock:
                    if self._future._task_state == CANCELLED:
                        raise CancelledError()

                    # run tiled acquisition for the selected roi
                    logging.debug(f"Z-stack acquisition is running for roi number {idx} with {roi} values")

                    self._future.running_subf = acquireTiledArea(streams=self.streams,
                                                                 stage=self._stage,
                                                                 area=roi,
                                                                 focus_range=self.focus_rng,
                                                                 overlap=self._overlap,
                                                                 settings_obs=self._settings_obs,
                                                                 log_path=self._log_path,
                                                                 zlevels=self._zlevels,
                                                                 registrar=self._registrar,
                                                                 weaver=self._weaver,
                                                                 focusing_method=self.focusing_method,
                                                                 focus_points=focus_points)
                    # remove the current tiled acquisition time from the list and assign the remaining time to the future
                    # along with the time update of the current tiled acquisition sub future
                    self.time_per_task.pop(0)
                    self._future.running_subf.add_update_callback(self._pass_future_progress)

                try:
                    da = self._future.running_subf.result()
                    # append all of them, when multiple streams are acquired
                    da_rois.extend(da)
                except Exception:
                    logging.debug(
                        f"Z-stack acquisition within roi failed for roi number {idx} with {roi}")
                    raise

                logging.debug(f"acquisition overview is completed for roi number {idx} with {roi} values")

                # cancel the sub future
                with self._future._task_lock:
                    if self._future._task_state == CANCELLED:
                        raise CancelledError()

        except CancelledError:
            logging.debug("Stopping because acquisition overview was cancelled")
            raise
        except Exception:
            logging.exception("acquisition overview failed")
            self._future.running_subf.cancel()
            raise
        finally:
            # state that the future has finished
            with self._future._task_lock:
                self._future._task_state = FINISHED

        return da_rois

def get_stream_based_bbox(
    pos: Dict[str, float],
    streams: List[Stream],
    tiles_nx: int,
    tiles_ny: int,
    overlap: float,
    tiling_rng: Dict[str, list],  # TODO: make optional, use axes range otherwise
) -> Optional[Tuple[float]]:
    """
    Compute a bounding box based on streams, based on the number of x tiles, stream fov and overlap.
    :param pos: the current position of the stage
    :param streams: the streams to acquire
    :param tiles_nx: the number of tiles in the x direction
    :param tiles_ny: the number of tiles in the y direction
    :param overlap: the overlap between tiles (in percentage)
    :param tiling_rng: the tiling range along x and y axes as (xmin, ymin, xmax, ymax), or (xmin, ymax, xmax, ymin)
    :return: the bounding box (xmin, ymin, xmax, ymax in physical coordinates). If no area at all, returns None.
    """
    # Get al stream fov's, if any
    fovs = [get_fov(s) for s in streams]
    if not fovs:
        # fall back to a small fov (default)
        fov = DEFAULT_FOV
    else:
        # smallest fov
        fov = tuple(map(min, zip(*fovs)))

    return get_fov_based_bbox(
        pos=pos,
        fov=fov,
        tiles_nx=tiles_nx,
        tiles_ny=tiles_ny,
        tiling_rng=tiling_rng,
        overlap=overlap,
    )


def get_tiled_bboxes(
    rel_bbox: Tuple[float],
    sample_centers: List[Tuple[float]],
) -> List[Tuple[float]]:
    """
    Compute bounding boxes for a given relative bounding box to different sample centers
    :param rel_bbox: the relative bounding box of the sample grid (xmin, ymin, xmax, ymax)
    :param sample_centers: the centers of the sample grids
    return: the bounding boxes (xmin, ymin, xmax, ymax in physical coordinates)
    corresponding to the sample centers.
    """
    # TODO: for now this is only for the MIMAS, but eventually, on the METEOR,
    # which often supports 2 grids, this should also be possible to use.
    # TODO: ensure that sample centers are converted to same coordinate system as stage position

    # Sort the (selected) centers along X, and for centers with the same X, along the Y.
    # In theory, the order doesn't really matter (at best it would safe a few
    # seconds of stage movement), but it's nice for the user that acquisitions
    # are done always in the same order, as it reduces "astonishment".
    sorted_centers = sorted(sample_centers)
    bboxes = [
        (
            center[0] + rel_bbox[0],
            center[1] + rel_bbox[1],
            center[0] + rel_bbox[2],
            center[1] + rel_bbox[3],
        )
        for center in sorted_centers
    ]

    return bboxes

def get_fov_based_bbox(
    pos: Dict[str, float],
    fov: Tuple[float, float],
    tiles_nx: int,
    tiles_ny: int,
    tiling_rng: Dict[str, list],
    overlap: float,
) -> Optional[Tuple[float, float]]:
    """
    Calculates the requested bounding box, based on the number of tiles, fov and overlap.
    :param pos: the current position of the stage
    :param fov: the fov to acquire, default is DEFAULT_FOV
    :param tiles_nx: the number of tiles in the x direction
    :param tiles_ny: the number of tiles in the y direction
    :param tiling_rng: the tiling range along x and y axes as (xmin, ymin, xmax, ymax), or (xmin, ymax, xmax, ymin)
    :param overlap: the overlap between tiles (in percentage)
    :return: the bounding box (xmin, ymin, xmax, ymax in physical coordinates). If no area at all, returns None.
    """
    # these formulas for w and h have to match the ones used in the 'stitching' module.
    w = tiles_nx * fov[0] * (1 - overlap)
    h = tiles_ny * fov[1] * (1 - overlap)

    # Note the area can accept LTRB or LBRT.
    bbox = clip_tiling_bbox_to_range(w, h, pos, tiling_rng)
    if bbox is None:
        # there is no intersection
        logging.warning(
            "Couldn't find intersection between stage pos %s and tiling range %s"
            % (pos, tiling_rng)
        )

    return bbox

def get_fov(s: Stream):
    """Get the field of view of a stream"""
    try:
        return s.guessFoV()
    except (NotImplementedError, AttributeError):
        raise TypeError("Unsupported Stream %s, it doesn't have a .guessFoV()" % (s,))

def clip_tiling_bbox_to_range(
    w: float, h: float, pos: Dict[str, float], tiling_rng: Dict[str, List[float]]
) -> Optional[Tuple[float]]:
    """
    Finds the intersection between the requested tiling bounding box and the tiling range.
    :param w: the width of the tiling bounding box
    :param h: the height of the tiling bounding box
    :param pos: the current position of the stage
    :param tiling_rng: the maximum tiling range along x and y axes as
            (xmin, ymin, xmax, ymax), or (xmin, ymax, xmax, ymin). the maximum range is
            specified by the task, and is usually the stage axes range or the imaging range.
    :returns: (None or tuple of 4 floats): None if there is no intersection, or
        the rectangle representing the intersection as (xmin, ymin, xmax, ymax).
    """
    bbox_req = (pos["x"] - w / 2, pos["y"] - h / 2, pos["x"] + w / 2, pos["y"] + h / 2)
    # clip the tiling area, if needed (or find the intersection between the active range and the requested area)
    return rect_intersect(
        bbox_req,
        (
            tiling_rng["x"][0],
            tiling_rng["y"][1],
            tiling_rng["x"][1],
            tiling_rng["y"][0],
        ),
    )

def get_zstack_levels(zsteps: int, zstep_size: float, rel: bool = False, focuser=None):
    """
    Calculate the zstack levels from the current focus position and zsteps value
    :param zsteps: (int) the number of zsteps
    :param zstep_size: (float) the size of each zstep
    :param rel: If this is False (default), then z stack levels are in absolute values. If rel is set to True then
        the z stack levels are calculated relative to each other.
    :param focuser: (Actuator) the focuser to use for the zstack levels
    :returns:
        (list(float) or None) zstack levels for zstack acquisition. None if only one zstep is requested.
    """
    # TODO: consolidate with odemis.util.comp.generate_zlevels
    # TODO: fix corner cases for shifting the zlevels

    if zsteps == 1:
        return None

    if not rel and focuser is None:
        raise ValueError("A focuser is required to calculate absolute zstack levels.")

    if rel:
        zmin = -(zsteps / 2 * zstep_size)
        zmax = +(zsteps / 2 * zstep_size)
    else:
        # Clip zsteps value to allowed range
        focus_value = focuser.position.value["z"]
        focus_range = focuser.axes["z"].range
        zmin = focus_value - (zsteps / 2 * zstep_size)
        zmax = focus_value + (zsteps / 2 * zstep_size)
        if (zmax - zmin) > (focus_range[1] - focus_range[0]):
            # Corner case: it'd be larger than the entire range => limit to the entire range
            zmin = focus_range[0]
            zmax = focus_range[1]
        if zmax > focus_range[1]: # FIXME: this can shift the focus outside of the range
            # Too high => shift down
            shift = zmax - focus_range[1]
            zmin -= shift
            zmax -= shift

        if zmin < focus_range[0]: # FIXME: this can shift the focus outside of the range
            # Too low => shift up
            shift = focus_range[0] - zmin
            zmin += shift
            zmax += shift

    # TODO: if there is an even number of zsteps, the current focus position will not be part of the zlevels.
    # As the current focus position can often be the "best" position, we could change the algorithm in such case to
    # shift exactly symmetrical, but instead use the current position as one of the zlevels.

    # Create focus zlevels from the given zsteps number
    zlevels = numpy.linspace(zmin, zmax, zsteps).tolist()

    return zlevels
