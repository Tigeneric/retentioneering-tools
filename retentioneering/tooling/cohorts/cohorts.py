from __future__ import annotations

from typing import Literal, Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from retentioneering.backend.tracker import (
    collect_data_performance,
    time_performance,
    track,
)
from retentioneering.constants import DATETIME_UNITS, DATETIME_UNITS_LIST
from retentioneering.eventstream.types import EventstreamType

# @TODO Подумать над сокращением списка поддерживаемых типов для когорт? dpanina


class Cohorts:
    """
    A class that provides methods for cohort analysis. The users are split into groups
    depending on the time of their first appearance in the eventstream; thus each user is
    associated with some ``cohort_group``. Retention rates of the active users
    belonging to each ``cohort_group`` are  calculated within each ``cohort_period``.

    Parameters
    ----------
    eventstream : EventstreamType


    See Also
    --------
    .Eventstream.cohorts : Call Cohorts tool as an eventstream method.
    .EventTimestampHist : Plot the distribution of events over time.
    .UserLifetimeHist : Plot the distribution of user lifetimes.

    Notes
    -----
    See :doc:`Cohorts user guide</user_guides/cohorts>` for the details.

    """

    __eventstream: EventstreamType
    cohort_start_unit: DATETIME_UNITS
    cohort_period: int
    cohort_period_unit: DATETIME_UNITS

    average: bool
    cut_bottom: int
    cut_right: int
    cut_diagonal: int
    start_event: str
    variant: str
    _cohort_matrix_result: pd.DataFrame

    @time_performance(
        scope="cohorts",
        event_name="init",
    )
    def __init__(self, eventstream: EventstreamType):
        self.active_size = pd.DataFrame()
        self.__eventstream = eventstream
        self.user_col = self.__eventstream.schema.user_id
        self.event_col = self.__eventstream.schema.event_name
        self.time_col = self.__eventstream.schema.event_timestamp

        self._cohort_matrix_result = pd.DataFrame()
        self.group_size = pd.DataFrame()
        self.unique_values = {}
    '''

    @staticmethod
    def _get_min_max_dates(data: pd.Series, freq) -> tuple:
        start_point = np.datetime64(data.min().to_period(freq).start_time, freq)
        end_point = np.datetime64(data.max(), freq) #+ step
        return start_point, end_point

    @staticmethod
    def _create_cohorts(min_date: pd.Timestamp, max_date: pd.Timestamp, freq: str, cohort_period: int, cohort_period_unit: str) -> pd.DataFrame:
        step = np.timedelta64(cohort_period, cohort_period_unit)

        # Generating an array of cohort group start dates
        # The range starts from the earliest user's event date (start_point) and ends at the latest user's event date (end_point),
        # with each step representing one cohort period (step)
        coh_groups_start_dates = np.arange(min_date, max_date, step)

        # Converting the array of cohort start dates to datetime format, and then converting to periods
        # The period format is based on the previously defined frequency (freq)
        coh_groups_start_dates = pd.to_datetime(coh_groups_start_dates).to_period(freq)

        if max_date < coh_groups_start_dates[-1].start_time:
            coh_groups_start_dates = coh_groups_start_dates[:-1]
        cohorts_list = pd.DataFrame(data=coh_groups_start_dates, columns=["CohortGroup"])
        cohorts_list["CohortGroupNum"] = np.arange(1, len(cohorts_list) + 1)
        return cohorts_list

    def _add_cohort_info(self, data: pd.DataFrame, cohorts_list: pd.DataFrame, freq: str) -> pd.DataFrame:
        data["OrderPeriod"] = pd.to_datetime(data[self.time_col]).dt.to_period(freq)
        start_int = cohorts_list["CohortGroup"].min().ordinal
        converter_freq = np.timedelta64(cohorts_list["CohortGroup"].freq.n, cohorts_list["CohortGroup"].freq.name[0])
        converter_freq_ = converter_freq.astype(f"timedelta64[{freq}]").astype(int)
        data["CohortGroupNum"] = (data["user_min_date_gr"].view("int64") - start_int + converter_freq_) // converter_freq_
        data = data.merge(cohorts_list, on="CohortGroupNum", how="left")
        data["CohortPeriod"] = ((data["OrderPeriod"].view("int64") - (data["CohortGroup"].view("int64") + converter_freq_)) // converter_freq_) + 1
        return data

    def add_cohort_analysis_data(
        self,
        data: pd.DataFrame,
        cohort_start_unit: DATETIME_UNITS,
        cohort_period: int,
        cohort_period_unit: DATETIME_UNITS,
    ) -> pd.DataFrame:

        data = data.copy()

        data["user_min_date_gr"] = self._calculate_user_min_dates(data, self.user_col, self.time_col)

        freq = self._adjust_frequency(cohort_start_unit, cohort_period_unit)

        min_cohort_date, max_cohort_date = self._get_min_max_dates(data["user_min_date_gr"], freq)

        data["user_min_date_gr"] = pd.to_datetime(data["user_min_date_gr"]).dt.to_period(freq)

        cohorts_list = self._create_cohorts(min_cohort_date, max_cohort_date, freq, cohort_period, cohort_period_unit)

        data = self._add_cohort_info(data, cohorts_list, freq)

        return data


    '''


    @staticmethod
    def _calculate_user_min_dates(data: pd.DataFrame, user_col: str, time_col: str) -> pd.Series:
        return data.groupby(user_col)[time_col].transform('min')

    @staticmethod
    def _adjust_frequency(cohort_start_unit: str, cohort_period_unit: str) -> str:
        if cohort_start_unit == "W":
            return "D"
        if DATETIME_UNITS_LIST.index(cohort_start_unit) >= DATETIME_UNITS_LIST.index(cohort_period_unit):
            return cohort_start_unit
        return cohort_period_unit

    def _cumulative_unique_users(self, row):
        cohort_group = row['CohortGroup']
        user = row[self.user_col]

        if cohort_group not in self.unique_values:
            self.unique_values[cohort_group] = set()

        self.unique_values[cohort_group].add(user)
        return len(self.unique_values[cohort_group])

    def add_cohort_analysis_data(
        self,
        data: pd.DataFrame,
        cohort_start_unit: DATETIME_UNITS,
        cohort_period: int,
        cohort_period_unit: DATETIME_UNITS,
    ) -> pd.DataFrame:

        #data = data.copy()

        freq = self._adjust_frequency(cohort_start_unit, cohort_period_unit)

        # Setting the frequency for grouping users into cohorts based on the starting unit of the cohort
        #freq = cohort_start_unit

        # Calculating the minimum event date for each user
        # This will be used to determine the user's cohort
        data["user_min_date_gr"] = self._calculate_user_min_dates(data, self.user_col, self.time_col)

        # Finding the minimum and maximum dates among all users
        # This will be used to define the time range for the cohorts
        min_cohort_date = data["user_min_date_gr"].min().to_period(cohort_start_unit).start_time
        max_cohort_date = data["user_min_date_gr"].max()

        # Checking if we need to change the time frequency for cohort grouping
        # This may be necessary if the time unit for the start of the cohort is smaller than the time unit for the cohort period
        #if DATETIME_UNITS_LIST.index(cohort_start_unit) < DATETIME_UNITS_LIST.index(cohort_period_unit):
        #    freq = cohort_period_unit

        # Changing the time frequency to 'D' (days) if it was set to 'W' (weeks)
        # This is done to ensure more accurate grouping
        #if freq == "W":
        #    freq = "D"

        # Converting the minimum event date of each user to a period based on the time frequency
        data["user_min_date_gr"] = data["user_min_date_gr"].dt.to_period(freq)

        # Calculating the time step to define the boundaries of the cohorts
        # For example, if the cohort period is 1 month, the step will be equal to 1 month
        step = np.timedelta64(cohort_period, cohort_period_unit)

        # Converting the minimum and maximum dates to the numpy datetime64 format for further calculations
        start_point = np.datetime64(min_cohort_date, freq)
        end_point = np.datetime64(max_cohort_date, freq) + step

        # Generating an array of cohort group start dates
        # The range starts from the earliest user's event date (start_point) and ends at the latest user's event date (end_point),
        # with each step representing one cohort period (step)
        coh_groups_start_dates = np.arange(start_point, end_point, step)

        # Converting the array of cohort start dates to datetime format, and then converting to periods
        # The period format is based on the previously defined frequency (freq)
        coh_groups_start_dates = pd.to_datetime(coh_groups_start_dates).to_period(freq)

        # Checking if the last cohort group's start date is later than the latest user's event date
        # If true, this means the last cohort group is empty and should be removed
        if max_cohort_date < coh_groups_start_dates[-1].start_time:
            # Remove the last cohort group start date from the array
            coh_groups_start_dates = coh_groups_start_dates[:-1]

        #old version
        #cohorts_list = pd.DataFrame(
        #    data=coh_groups_start_dates, index=None, columns=["CohortGroup"]  # type: ignore
        #).reset_index()
        #cohorts_list.columns = ["CohortGroupNum", "CohortGroup"]  # type: ignore
        #cohorts_list["CohortGroupNum"] += 1

        cohorts_list = pd.DataFrame(data=coh_groups_start_dates, columns=["CohortGroup"])
        cohorts_list["CohortGroupNum"] = np.arange(1, len(cohorts_list) + 1)

        data["OrderPeriod"] = data[self.time_col].dt.to_period(freq)
        start_int = pd.Series(min_cohort_date.to_period(freq=freq)).astype(int)[0]

        converter_freq = np.timedelta64(cohort_period, cohort_period_unit)
        converter_freq_ = converter_freq.astype(f"timedelta64[{freq}]").astype(int)
        data["CohortGroupNum"] = (data["user_min_date_gr"].astype(int) - start_int + converter_freq_) // converter_freq_

        data = data.merge(cohorts_list, on="CohortGroupNum", how="left")

        data["CohortPeriod"] = (
            (data["OrderPeriod"].astype(int) - (data["CohortGroup"].astype(int) + converter_freq_))  # type: ignore
            // converter_freq_
        ) + 1

        data = data.sort_values(by=["CohortGroup", "CohortPeriod", self.user_col])
        data['cumulative_unique_users'] = data.apply(self._cumulative_unique_users, axis=1)

        #data.to_csv('test.csv')
        return data

    def _changed_variant(self, cohorts, variant):
        if variant == 'classic':
            new_col_name = self.user_col
        elif variant == 'returned':
            new_col_name = 'max_value_col'
        else:
            # Если variant не соответствует ни одному из ожидаемых значений,
            # вы можете выбросить исключение или обработать этот случай по-другому.
            raise ValueError(f'Неизвестный вариант: {variant}')

        cohorts.rename(columns={new_col_name: "TotalUsers"}, inplace=True)
        return cohorts

    @staticmethod
    def _cut_cohort_matrix(
        df: pd.DataFrame, cut_bottom: int = 0, cut_right: int = 0, cut_diagonal: int = 0
    ) -> pd.DataFrame:
        for row in df.index:
            df.loc[row, max(0, df.loc[row].notna()[::-1].idxmax() + 1 - cut_diagonal) :] = None  # type: ignore

        return df.iloc[: len(df) - cut_bottom, : len(df.columns) - cut_right]

    @time_performance(
        scope="cohorts",
        event_name="fit",
    )
    def fit(
        self,
        cohort_start_unit: DATETIME_UNITS,
        cohort_period: Tuple[int, DATETIME_UNITS],
        average: bool = True,
        cut_bottom: int = 0,
        cut_right: int = 0,
        cut_diagonal: int = 0,
        start_event: str = None,
        variant: str = None
    ) -> None:
        """
        Calculates the cohort internal values with the defined parameters.
        Applying ``fit`` method is necessary for the following usage
        of any visualization or descriptive ``Cohorts`` methods.

        Parameters
        ----------
        start_event: .....
        cohort_start_unit : :numpy_link:`DATETIME_UNITS<>`
            The way of rounding and formatting of the moment from which the cohort count begins.
            The minimum timestamp is rounded down to the selected datetime unit.

            For example:
            assume we have an eventstream with the following minimum timestamp - "2021-12-28 09:08:34.432456".
            The result of roundings with different ``DATETIME_UNITS`` is shown in the table below:

            +------------------------+-------------------------+
            | **cohort_start_unit**  | **cohort_start_moment** |
            +------------------------+-------------------------+
            | Y                      |  2021-01-01 00:00:00    |
            +------------------------+-------------------------+
            | M                      |  2021-12-01 00:00:00    |
            +------------------------+-------------------------+
            | W                      |  2021-12-27 00:00:00    |
            +------------------------+-------------------------+
            | D                      |  2021-08-28 00:00:00    |
            +------------------------+-------------------------+

        cohort_period : Tuple(int, :numpy_link:`DATETIME_UNITS<>`)
            The cohort_period size and its ``DATETIME_UNIT``. This parameter is used in calculating:

            - Start moments for each cohort from the moment specified with the ``cohort_start_unit`` parameter
            - Cohort periods for each cohort from its start moment.
        average : bool, default True
            - If ``True`` - calculating average for each cohort period.
            - If ``False`` - averaged values aren't calculated.
        cut_bottom : int, default 0
            Drop 'n' rows from the bottom of the cohort matrix.
            Average is recalculated.
        cut_right : int, default 0
            Drop 'n' columns from the right side of the cohort matrix.
            Average is recalculated.
        cut_diagonal : int, default 0
            Replace values in 'n' diagonals (last period-group cells) with ``np.nan``.
            Average is recalculated.

        Notes
        -----
        Parameters ``cohort_start_unit`` and ``cohort_period`` should be consistent.
        Due to "Y" and "M" being non-fixed types, it can be used only with each other
        or if ``cohort_period_unit`` is more detailed than ``cohort_start_unit``.
        More information - :numpy_timedelta_link:`about numpy timedelta<>`

        Only cohorts with at least 1 user in some period are shown.

        See :doc:`Cohorts user guide</user_guides/cohorts>` for the details.

        """
        called_params = {
            "cohort_start_unit": cohort_start_unit,
            "cohort_period": cohort_period,
            "average": average,
            "cut_bottom": cut_bottom,
            "cut_right": cut_right,
            "cut_diagonal": cut_diagonal,
        }
        not_hash_values = ["cohort_start_unit", "cohort_period"]

        data = self.__eventstream.to_dataframe()
        self.average = average
        self.cohort_start_unit = cohort_start_unit
        self.cohort_period, self.cohort_period_unit = cohort_period
        self.cut_bottom = cut_bottom
        self.cut_right = cut_right
        self.cut_diagonal = cut_diagonal
        self.start_event = start_event
        self.variant = variant

        if self.cohort_period <= 0:
            raise ValueError("cohort_period should be positive integer!")

        # @TODO добавить ссылку на numpy с объяснением. dpanina
        if self.cohort_period_unit in ["Y", "M"] and self.cohort_start_unit not in ["Y", "M"]:
            raise ValueError(
                """Parameters ``cohort_start_unit`` and ``cohort_period`` should be consistent.
                                 Due to "Y" and "M" are non-fixed types it can be used only with each other
                                 or if ``cohort_period_unit`` is more detailed than ``cohort_start_unit``!"""
            )

        df = self.add_cohort_analysis_data(
            data=data,
            cohort_start_unit=self.cohort_start_unit,
            cohort_period=self.cohort_period,
            cohort_period_unit=self.cohort_period_unit,
        )

        cohorts = df[df['event'] != start_event].groupby(["CohortGroup", "CohortPeriod"]).agg(
                user_id=(self.user_col, lambda x: x.nunique()),
                max_value_col=('cumulative_unique_users', 'max')
            ).reset_index()
        #print(cohorts)
        #print(cohorts.columns)
        total_by_cohorts = df[df['event'] != start_event].groupby(["CohortGroup"])[[self.user_col]].nunique()

        self._changed_variant(cohorts, self.variant)
        #cohorts.rename(columns={self.user_col: "TotalUsers"}, inplace=True)
        #cohorts.rename(columns={"max_value_col": "TotalUsers"}, inplace=True)
        cohorts.set_index(["CohortGroup", "CohortPeriod"], inplace=True)
        cohort_group_size = df.groupby(["CohortGroup", "CohortPeriod"])[[self.user_col]].nunique()
        cohort_group_size = cohort_group_size[self.user_col].groupby(level=0).first()
        #cohort_group_size = cohorts["TotalUsers"].groupby(level=0).first()
        cohorts.reset_index(inplace=True)
        user_retention = (
            cohorts.pivot(index="CohortPeriod", columns="CohortGroup", values="TotalUsers").divide(
                cohort_group_size, axis=1
            #cohorts["TotalUsers"].unstack(0).divide(cohort_group_size, axis=1)
            )
        ).T

        user_retention = self._cut_cohort_matrix(
            df=user_retention, cut_diagonal=self.cut_diagonal, cut_bottom=self.cut_bottom, cut_right=self.cut_right
        )
        user_retention.index = user_retention.index.astype(str)
        if self.average:
            user_retention.loc["Average"] = user_retention.mean()

        self._cohort_matrix_result = user_retention
        self.group_size = cohort_group_size.to_frame(name="TotalUsers")
        self.active_size = total_by_cohorts
        collect_data_performance(
            scope="cohorts",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={"shape": self._cohort_matrix_result.shape},
            eventstream_index=self.__eventstream._eventstream_index,
        )

    @time_performance(
        scope="cohorts",
        event_name="heatmap",
    )
    def heatmap(self, width: float = 5.0, height: float = 5.0) -> matplotlib.axes.Axes:
        """
        Builds a heatmap based on the calculated cohort matrix values.
        Should be used after :py:func:`fit`.

        Parameters
        ----------
        width : float, default 5.0
            Width of the figure in inches.
        height : float, default 5.0
            Height of the figure in inches.


        Returns
        -------
        matplotlib.axes.Axes
        """
        called_params = {
            "width": width,
            "height": height,
        }

        width_ratio_users_column = 1
        width_ratio_retention_column = 10
        df = self.group_size
        df2 = self._cohort_matrix_result
        df3 = (self.active_size['user_id'] / self.group_size['TotalUsers']).to_frame(name="ActiveShare")
        figsize = (width, height)
        fig, (ax1, ax2, ax3) = plt.subplots(figsize=figsize, ncols=3, gridspec_kw={'width_ratios':
            [width_ratio_users_column, width_ratio_retention_column, width_ratio_users_column]})
        sns.heatmap(df, annot=True, fmt=".0f", linewidths=1, linecolor="gray", cbar=False, ax=ax1)
        sns.heatmap(df2, annot=True, fmt=".1%", linewidths=1, yticklabels=False, linecolor="gray", ax=ax2)
        sns.heatmap(df3, annot=True, fmt=".1%", linewidths=1, yticklabels=False,  cbar=False, linecolor="gray", ax=ax3)
        ax2.set_ylabel('')
        ax3.set_ylabel('')
        plt.tight_layout()

        collect_data_performance(
            scope="cohorts",
            event_name="metadata",
            called_params=called_params,
            performance_data={"shape": self._cohort_matrix_result.shape},
            eventstream_index=self.__eventstream._eventstream_index,
        )

        return [ax1, ax2]

    @time_performance(
        scope="cohorts",
        event_name="lineplot",
    )
    def lineplot(
        self, plot_type: Literal["cohorts", "average", "all"] = "cohorts", width: float = 7.0, height: float = 5.0
    ) -> matplotlib.axes.Axes:
        """
        Create a chart representing each cohort dynamics over time.
        Should be used after :py:func:`fit`.

        Parameters
        ----------
        plot_type: 'cohorts', 'average' or 'all'
            - if ``cohorts`` - shows a lineplot for each cohort,
            - if ``average`` - shows a lineplot only for the average values over all the cohorts,
            - if ``all`` - shows a lineplot for each cohort and also for their average values.
        width : float, default 7.0
            Width of the figure in inches.
        height : float, default 5.0
            Height of the figure in inches.

        Returns
        -------
        matplotlib.axes.Axes

        """
        if plot_type not in ["cohorts", "average", "all"]:
            raise ValueError("plot_type parameter should be 'cohorts', 'average' or 'all'!")
        called_params = {
            "plot_type": plot_type,
            "width": width,
            "height": height,
        }
        not_hash_values = ["plot_type"]

        figsize = (width, height)
        df_matrix = self._cohort_matrix_result
        df_wo_average = df_matrix[df_matrix.index != "Average"]  # type: ignore
        if plot_type in ["all", "average"] and "Average" not in df_matrix.index:  # type: ignore
            df_matrix.loc["Average"] = df_matrix.mean()  # type: ignore
        df_average = df_matrix[df_matrix.index == "Average"]  # type: ignore
        figure, ax = plt.subplots(figsize=figsize)
        if plot_type == "all":
            sns.lineplot(df_wo_average.T, lw=1.5, ax=ax)
            sns.lineplot(df_average.T, lw=2.5, palette=["red"], marker="X", markersize=8, alpha=0.6, ax=ax)

        if plot_type == "average":
            sns.lineplot(df_average.T, lw=2.5, palette=["red"], marker="X", markersize=8, alpha=0.6, ax=ax)

        if plot_type == "cohorts":
            sns.lineplot(df_wo_average.T, lw=1.5, ax=ax)

        ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
        ax.set_xlabel("Period from the start of observation")
        ax.set_ylabel("Share of active users")

        collect_data_performance(
            scope="cohorts",
            event_name="metadata",
            called_params=called_params,
            not_hash_values=not_hash_values,
            performance_data={"shape": self._cohort_matrix_result.shape},
            eventstream_index=self.__eventstream._eventstream_index,
        )
        return ax

    @property
    @time_performance(
        scope="cohorts",
        event_name="values",
    )
    def values(self) -> pd.DataFrame:
        """
        Returns a pd.DataFrame representing the calculated cohort matrix values.
        Should be used after :py:func:`fit`.

        Returns
        -------
        pd.DataFrame

        """
        return self._cohort_matrix_result

    @property
    @time_performance(  # type: ignore
        scope="cohorts",
        event_name="params",
    )
    def params(self) -> dict[str, DATETIME_UNITS | tuple | bool | int | None]:
        """
        Returns the parameters used for the last fitting.
        Should be used after :py:func:`fit`.

        """
        return {
            "cohort_start_unit": self.cohort_start_unit,
            "cohort_period": (self.cohort_period, self.cohort_period_unit),
            "average": self.average,
            "cut_bottom": self.cut_bottom,
            "cut_right": self.cut_right,
            "cut_diagonal": self.cut_diagonal,
        }
