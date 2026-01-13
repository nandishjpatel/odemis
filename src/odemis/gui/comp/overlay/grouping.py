# -*- coding: utf-8 -*-


"""
:created: 2014-01-25
:author: Rinze de Laat
:copyright: © 2014-2021 Rinze de Laat, Éric Piel, Philip Winkler, Delmic

This file is part of Odemis.

.. license::
    Odemis is free software: you can redistribute it and/or modify it under the
    terms of the GNU General Public License version 2 as published by the Free
    Software Foundation.

    Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
    WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
    PARTICULAR PURPOSE. See the GNU General Public License for more details.

    You should have received a copy of the GNU General Public License along with
    Odemis. If not, see http://www.gnu.org/licenses/.

"""

import numpy
import logging
import odemis.gui as gui
from odemis.gui.model import TOOL_NONE
from odemis.gui.comp.overlay.base import (
    WorldOverlay,
)
from odemis.gui.comp.overlay.world_select import WorldSelectOverlay


class GroupingOverlay(WorldSelectOverlay):
    """Overlay representing one region of calibration (ROC) on the FastEM."""

    def __init__(
        self, cnvs, shapes_va, tool, tool_va, colour=gui.SELECTION_COLOUR,
    ):
        """
        cnvs (FastEMAcquisitionCanvas): canvas for the overlay
        coordinates (TupleContinuousVA): VA of 4 floats representing region of calibration coordinates
        label (str or int): label to be displayed next to rectangle
        sample_bbox (tuple): bounding box coordinates of the sample holder (minx, miny, maxx, maxy) [m]
        colour (str): hex colour code for ROC display in viewport
        """
        super().__init__(cnvs, colour)
        self.shapes = shapes_va
        self.tool = tool
        self._selected_tool_va = tool_va
        tool_va.subscribe(self._on_tool, init=True)

    def _on_tool(self, selected_tool):
        """Update the overlay when it's active and tools change."""
        self.active.value = selected_tool == self.tool

    def find_shapes_in_bbox_numpy(self, shapes, bbox):
        """
        Finds shapes with any point in a bounding box using NumPy for efficiency.

        Args:
            shapes: A list of shapes, where each shape is a list of (x, y) points.
            bbox: A tuple representing the bounding box (xmin, ymin, xmax, ymax).

        Returns:
            A list of shapes that have at least one point inside the bounding box.
        """
        xmin, ymin, xmax, ymax = bbox
        matching_shapes = []

        for shape in shapes:
            # Skip empty shapes that would cause an error in numpy.array
            if not shape:
                continue

            # 1. Convert the list of points to a NumPy array
            points_arr = numpy.array(shape.points.value) # Creates an array of shape (num_points, 2)

            # 2. Perform vectorized boolean comparison
            #    points_arr[:, 0] is the column of all x-coordinates
            #    points_arr[:, 1] is the column of all y-coordinates
            #    The '&' operator performs an element-wise logical AND.
            is_inside = (
                (points_arr[:, 0] >= xmin) & (points_arr[:, 0] <= xmax) &
                (points_arr[:, 1] >= ymin) & (points_arr[:, 1] <= ymax)
            )

            # 3. Use numpy.any() to check if any point was inside the box
            #    This is much faster than a Python loop.
            if numpy.any(is_inside):
                matching_shapes.append(shape)

        return matching_shapes

    def on_left_up(self, evt):
        if self.active.value:
            # Select region if clicked
            self._phys_to_view()
            rect = self.get_physical_sel()
            if rect:
                matching_shapes = self.find_shapes_in_bbox_numpy(self.shapes.value, rect)
                for shape in matching_shapes:
                    logging.debug(f"Shape {shape.name.value}")

            self.clear_selection()
            self.cnvs.update_drawing()  # Line width changes in .draw when .active is changed
            self.cnvs.reset_default_cursor()
            self._selected_tool_va.value = TOOL_NONE
        WorldOverlay.on_left_up(self, evt)

    def draw(self, ctx, shift=(0, 0), scale=1.0):
        """
        Draw with adaptive line width (depending on whether or not the overlay is active and enabled) and add label.
        """
        WorldSelectOverlay.draw(self, ctx, shift, scale, dash=True)
