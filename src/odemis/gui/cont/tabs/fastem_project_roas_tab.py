# -*- coding: utf-8 -*-

"""
@author: Nandish Patel

Copyright © 2024 Nandish Patel, Delmic

This file is part of Odemis.

Odemis is free software: you can redistribute it and/or modify it under the
terms of the GNU General Public License version 2 as published by the Free
Software Foundation.

Odemis is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with
Odemis. If not, see http://www.gnu.org/licenses/.

"""
import math
from enum import Enum

import wx
from wx.grid import (
    EVT_GRID_CELL_CHANGING,
    GridCellBoolEditor,
    GridCellFloatEditor,
    GridCellNumberEditor,
)

import odemis.gui.model as guimod
from odemis import model
from odemis.acq.fastem import FastEMROA
from odemis.gui.cont.fastem_grid_base import (
    DEFAULT_PARENT,
    EVT_GRID_ROW_CHANGED,
    Column,
    DynamicGridCellComboBoxEditor,
    GridBase,
    Row,
)
from odemis.gui.cont.tabs.fastem_project_sections_tab import SectionColumnNames
from odemis.gui.cont.tabs.tab import Tab
from odemis.gui.util import call_in_wx_main
from odemis.util import units


class ROAColumnNames(Enum):
    NAME = "Name"
    SLICE_IDX = "Slice Index"
    POSX = "Position.X [m]"
    POSY = "Position.Y [m]"
    SIZEX = "Size.X"
    SIZEY = "Size.Y"
    ROT = "Rotation [°]"
    PARENT = "Parent"
    FIELDS = "Show Fields"


class ROARow(Row):
    """Class representing a row in the ROA (Region of Interest) grid."""

    def __init__(self, data, roa, index=-1):
        super().__init__(data, roa, index)
        self.parent_name = model.StringVA(self.data[ROAColumnNames.PARENT.value])

    def to_dict(self):
        """
        Converts the ROARow instance to a dictionary format.

        :return: (dict) The ROARow data in dictionary format.
        """
        return {
            "index": self.index,
            "data": self.data,
            "roa": (
                self.roa.to_dict() if hasattr(self.roa, "to_dict") else str(self.roa)
            ),  # Assuming roa has a to_dict method or can be stringified
            "parent_name": self.parent_name.value,
        }

    @staticmethod
    def from_dict(roa: dict, tab_data):
        """
        Creates a ROARow instance from a dictionary.

        :param roa: (dict) Dictionary containing the ROA row data.
        :param tab_data: (MicroscopyGUIData) Data related to the tab.
        :return: (ROARow) The created ROARow instance.
        """
        index = int(roa["index"])
        data = roa["data"]
        parent_name = roa["parent_name"]
        roa = FastEMROA.from_dict(roa["roa"], tab_data)
        roa_row = ROARow(data, roa, index)
        roa_row.parent_name.value = parent_name
        return roa_row

    @staticmethod
    def find_next_slice_index(name, rows):
        """
        Finds the next available index for the given name in the list of rows.

        :param name: (str) The name for which the index is to be found.
        :param rows: (list) The list of existing rows.
        :return: (int) The next available index for the given name.
        """
        used_indices = {
            int(row.data[ROAColumnNames.SLICE_IDX.value])
            for row in rows
            if row.data[ROAColumnNames.NAME.value] == name
        }
        # Find the smallest missing index
        next_index = 0
        while next_index in used_indices:
            next_index += 1
        return next_index

    @staticmethod
    def is_unique_name_slice_idx(name, slice_idx, rows):
        """
        Checks if the given index is unique for the specified name in the list of rows.

        :param name: (str) The name to check for uniqueness.
        :param slice_idx: (int) The index to check for uniqueness.
        :param rows: (list) The list of existing rows.
        :return: (bool) True if the index is unique for the given name, False otherwise.
        """
        for row in rows:
            if (
                row.data[ROAColumnNames.NAME.value] == name
                and row.data[ROAColumnNames.SLICE_IDX.value] == slice_idx
            ):
                return False
        return True

    @staticmethod
    def set_grid_fill(flag: bool, col, grid):
        """
        Sets the grid fill flag for all rows in the grid.

        :param flag: (bool) Whether to fill the grid or not.
        :param col: (Column) The column that is affected.
        :param grid: (GridBase) The grid instance that contains the rows.
        """
        for row in grid.rows:
            row.data[ROAColumnNames.FIELDS.value] = "1" if flag else ""
            grid.SetCellValue(row.index, col.index, "1" if flag else "")
            row.roa.shape.fill_grid.value = flag

    def update_data(self):
        """
        Updates the row data based on the ROA.

        This method updates the position, size, rotation, and field data in the row
        based on the shape associated with the ROA. If the fields value is "1", it
        clears the fill grid flag and refreshes the drawing.
        """
        posx, posy = self.roa.shape.get_position()
        sizex, sizey = self.roa.shape.get_size()
        sizex = units.readable_str(sizex, unit="m", sig=3)
        sizey = units.readable_str(sizey, unit="m", sig=3)
        self.data[ROAColumnNames.POSX.value] = posx
        self.data[ROAColumnNames.POSY.value] = posy
        self.data[ROAColumnNames.SIZEX.value] = sizex
        self.data[ROAColumnNames.SIZEY.value] = sizey
        self.data[ROAColumnNames.ROT.value] = int(math.degrees(self.roa.shape.rotation))
        if self.data[ROAColumnNames.FIELDS.value] == "1":
            self.data[ROAColumnNames.FIELDS.value] = ""
            self.roa.shape.fill_grid.value = False
            self.roa.shape.cnvs.request_drawing_update()
            self.roa.shape.cnvs.Refresh()

    def on_cell_changing(self, new_value, row, col, grid):
        """
        Handles changes in cell values and updates associated data and ROA.

        :param new_value: (str) The new value to set in the cell.
        :param row: (int) The index of the row being modified.
        :param col: (int) The index of the column being modified.
        :param grid: (GridBase) The grid instance that contains the row.
        :return: (str) The value to set in the cell.
        """
        value_to_set = new_value
        rows = grid.rows
        col = grid.columns[col]
        if col.label == ROAColumnNames.NAME.value:
            current_slice_idx = self.data[ROAColumnNames.SLICE_IDX.value]
            if not ROARow.is_unique_name_slice_idx(
                value_to_set, current_slice_idx, rows
            ):
                new_slice_idx = ROARow.find_next_slice_index(value_to_set, rows)
                self.data[ROAColumnNames.SLICE_IDX.value] = int(new_slice_idx)
                self.roa.slice_index.value = new_slice_idx
                slice_index_col = grid.get_column_by_label(
                    ROAColumnNames.SLICE_IDX.value
                )
                grid.SetCellValue(row, slice_index_col.index, str(new_slice_idx))
            self.data[ROAColumnNames.NAME.value] = value_to_set
            self.roa.shape.name.value = (
                f"{value_to_set}_{self.data[ROAColumnNames.SLICE_IDX.value]}"
            )
            self.roa.name.value = value_to_set
        elif col.label == ROAColumnNames.SLICE_IDX.value:
            current_name = self.data[ROAColumnNames.NAME.value]
            if not ROARow.is_unique_name_slice_idx(
                current_name, int(value_to_set), rows
            ):
                new_slice_idx = ROARow.find_next_slice_index(current_name, rows)
                value_to_set = str(new_slice_idx)
            self.data[ROAColumnNames.SLICE_IDX.value] = int(value_to_set)
            self.roa.slice_index.value = int(value_to_set)
            self.roa.shape.name.value = (
                f"{self.data[ROAColumnNames.NAME.value]}_{value_to_set}"
            )
        elif col.label == ROAColumnNames.POSX.value:
            posy = self.data[ROAColumnNames.POSY.value]
            self.data[ROAColumnNames.POSX.value] = posx = float(value_to_set)
            self.roa.shape.move_to((posx, posy))
            self.roa.shape.cnvs.request_drawing_update()
        elif col.label == ROAColumnNames.POSY.value:
            posx = self.data[ROAColumnNames.POSX.value]
            self.data[ROAColumnNames.POSY.value] = posy = float(value_to_set)
            self.roa.shape.move_to((posx, posy))
            self.roa.shape.cnvs.request_drawing_update()
        elif col.label == ROAColumnNames.ROT.value:
            self.data[ROAColumnNames.ROT.value] = rot = int(value_to_set)
            self.roa.shape.set_rotation(math.radians(rot))
            self.roa.shape.cnvs.request_drawing_update()
        elif col.label == ROAColumnNames.PARENT.value:
            self.data[ROAColumnNames.PARENT.value] = value_to_set
            self.parent_name.value = value_to_set
        elif col.label == ROAColumnNames.FIELDS.value:
            ROARow.set_grid_fill(False, col, grid)
            self.data[ROAColumnNames.FIELDS.value] = value_to_set
            if value_to_set == "1":
                self.roa.shape.grid_rects = self.roa.field_rects
                self.roa.shape.fill_grid.value = True
                self.roa.shape.cnvs.request_drawing_update()
            else:
                self.roa.shape.fill_grid.value = False
                self.roa.shape.cnvs.request_drawing_update()
        return value_to_set


class FastEMProjectROAsTab(Tab):
    """Tab class for displaying and interacting with ROAs data for a project."""

    def __init__(self, name, button, panel, main_frame, main_data, sections_grid):

        self.tab_data = guimod.MicroscopyGUIData(main_data)
        super().__init__(name, button, panel, main_frame, self.tab_data)
        # Flag to indicate the tab has been fully initialized or not. Some initialisation
        # need to wait for the tab to be shown on the first time.
        self._initialized_after_show = False

        self.grid = GridBase(panel, size=panel.Parent.Size)
        self.columns = [
            Column(0, ROAColumnNames.NAME.value),
            Column(
                1,
                ROAColumnNames.SLICE_IDX.value,
                editor_cls=GridCellNumberEditor,
                editor_args={"min": 0, "max": 1000000},
            ),
            Column(
                2,
                ROAColumnNames.PARENT.value,
                editor_cls=DynamicGridCellComboBoxEditor,
                editor_args={"choices": [DEFAULT_PARENT]},
            ),
            Column(
                3,
                ROAColumnNames.POSX.value,
                editor_cls=GridCellFloatEditor,
                editor_args={"precision": 9},
            ),
            Column(
                4,
                ROAColumnNames.POSY.value,
                editor_cls=GridCellFloatEditor,
                editor_args={"precision": 9},
            ),
            Column(
                5,
                ROAColumnNames.SIZEX.value,
                editor_cls=GridCellFloatEditor,
                is_read_only=True,
            ),
            Column(
                6,
                ROAColumnNames.SIZEY.value,
                editor_cls=GridCellFloatEditor,
                is_read_only=True,
            ),
            Column(
                7,
                ROAColumnNames.ROT.value,
                editor_cls=GridCellNumberEditor,
                editor_args={"min": 0, "max": 360},
            ),
            Column(8, ROAColumnNames.FIELDS.value, editor_cls=GridCellBoolEditor),
        ]
        self.grid.set_columns(self.columns)

        self.sections_grid = sections_grid
        self.sections_grid.Bind(
            EVT_GRID_CELL_CHANGING, self._on_sections_grid_cell_changing
        )
        self.sections_grid.Bind(EVT_GRID_ROW_CHANGED, self._on_sections_row_changed)

        self.panel.Bind(wx.EVT_SIZE, self._on_panel_size)

    def _on_sections_grid_cell_changing(self, evt):
        """Handles changes in the sections grid cell values and updates the parent column in the ROA grid."""
        name_col = self.sections_grid.get_column_by_label(SectionColumnNames.NAME.value)
        col = evt.GetCol()
        row = evt.GetRow()
        if name_col.index == col:
            current_value = self.sections_grid.GetCellValue(row, col)
            new_value = evt.GetString()
            self._update_parent_col(current_value, new_value)
        evt.Skip()

    def _on_sections_row_changed(self, evt):
        """Updates the parent column in the ROA grid when the sections grid row changes."""
        self._update_parent_col()
        evt.Skip()

    @call_in_wx_main
    def _update_parent_col(self, current_value=None, new_value=None):
        """
        Updates the parent column values in the ROA grid based on section data.

        :param current_value: (str, optional) The current value of the parent column to update.
        :param new_value: (str, optional) The new value to set in the parent column.
        """
        section_names = self.sections_grid.get_column_values_by_label(
            SectionColumnNames.NAME.value
        )
        section_slice_index = self.sections_grid.get_column_values_by_label(
            SectionColumnNames.SLICE_IDX.value
        )
        section_info = [
            f"{name}_{index}" for name, index in zip(section_names, section_slice_index)
        ]
        section_info.append(DEFAULT_PARENT)
        parent_col = self.grid.get_column_by_label(SectionColumnNames.PARENT.value)
        parent_editors = self.grid.get_column_editors(parent_col.index)
        for row_idx, parent_editor in enumerate(parent_editors):
            parent_editor.SetParameters(",".join(sorted(section_info)))
            self.grid.SetCellEditor(row_idx, parent_col.index, parent_editor)
            current_parent = self.grid.GetCellValue(row_idx, parent_col.index)
            if current_value and new_value and current_parent == current_value:
                self.grid.SetCellValue(row_idx, parent_col.index, new_value)
                self.grid.rows[row_idx].parent_name.value = new_value
            elif current_parent not in section_info:
                self.grid.SetCellValue(row_idx, parent_col.index, DEFAULT_PARENT)
                self.grid.rows[row_idx].parent_name.value = DEFAULT_PARENT

    def _on_panel_size(self, evt):
        """Adjusts the grid size when the panel size changes."""
        self.grid.SetSize(self.panel.Parent.Size)
        self.grid.Layout()
        self.grid.Refresh()

    def Show(self, show=True):
        super().Show(show)

        if show and not self._initialized_after_show:
            self._initialized_after_show = True

    @classmethod
    def get_display_priority(cls, main_data):
        # Tab is used only for FastEM
        if main_data.role in ("mbsem",):
            return 4
        else:
            return None