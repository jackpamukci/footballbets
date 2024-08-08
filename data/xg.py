import math
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from tqdm import tqdm
import pandas as pd


class xG:
    def __init__(self, spadldata: pd.DataFrame):

        self.features_calculated = False
        self.model_trained = False
        self.X_h = None
        self.X_f = None
        self.spadldf = self._get_xg_features(spadldata)

    def get_xg(self):

        if self.features_calculated == False:
            raise ValueError("Features must be calculated first")

        if self.model_trained == False:
            self._train_model(self.spadldf)

        self.spadldf = self.spadldf.merge(
            self.X_h["xG"], how="left", left_index=True, right_index=True
        )
        self.spadldf = self.spadldf.merge(
            self.X_f["xG"], how="left", left_index=True, right_index=True
        )
        self.spadldf["xG"] = self.spadldf["xG_x"].combine_first(self.spadldf["xG_y"])
        self.spadldf.drop(["xG_x", "xG_y"], axis=1, inplace=True)

        self.spadldf.loc[self.spadldf.type_name == "shot_penalty", "xG"] = 0.76
        return self.spadldf.xG

    def _train_model(self, spadldf):
        scaler = MinMaxScaler()

        # Split into foot and header shots / Get Dummies / Normalize Values
        num_cols = [
            "xG_distance",
            "xG_angle",
            "xG_distance_inv",
            "xG_angle_inv",
            "dist_ang_inv",
        ]
        bool_cols = [
            "is_corner",
            "is_cross",
            "is_freekick",
            "is_cutback",
            "is_throughball",
            "after_dribble",
            "play_type",
            "goal",
        ]

        nor_shots = spadldf[
            (spadldf["type_name"].isin(["shot", "shot_freekick"]))
            & (spadldf["body"] == "foot")
        ][num_cols + bool_cols]
        nor_shots = pd.concat(
            [nor_shots, pd.get_dummies(nor_shots["play_type"])], axis=1
        ).drop("play_type", axis=1)

        headers = spadldf[
            (spadldf["type_name"] == "shot") & (spadldf["body"] == "header")
        ][num_cols + bool_cols]
        headers = pd.concat(
            [headers, pd.get_dummies(headers["play_type"])], axis=1
        ).drop("play_type", axis=1)
        bool_cols.remove("play_type")

        nor_shots[num_cols] = scaler.fit_transform(nor_shots[num_cols])
        nor_shots[bool_cols] = nor_shots[bool_cols].astype(int)

        headers[num_cols] = scaler.fit_transform(headers[num_cols])
        headers[bool_cols] = headers[bool_cols].astype(int)

        # Train Data

        X_f = nor_shots.drop("goal", axis=1)
        y_f = nor_shots["goal"]
        X_h = headers.drop("goal", axis=1)
        y_h = headers["goal"]

        head_model = LogisticRegression()
        foot_model = LogisticRegression()

        head_model.fit(X_h, y_h)
        foot_model.fit(X_f, y_f)

        y_pred_f = foot_model.predict_proba(X_f)[:, 1]
        y_pred_h = head_model.predict_proba(X_h)[:, 1]

        X_f["xG"] = y_pred_f
        X_h["xG"] = y_pred_h

        self.X_f = X_f
        self.X_h = X_h
        self.model_trained = True

    def _get_xg_features(self, spadldf):

        # xG Distance and Angle
        spadldf["xG_distance"] = np.sqrt(
            (spadldf["start_x"] - 105) ** 2 + (spadldf["start_y"] - 34) ** 2
        )
        spadldf["xG_angle"] = spadldf.apply(
            lambda x: self._calculate_relative_angle(x.start_x, x.start_y), axis=1
        )
        spadldf["xG_distance_inv"] = 1 / spadldf.xG_distance
        spadldf["xG_angle_inv"] = 1 / spadldf.xG_angle
        spadldf["dist_ang_inv"] = 1 / (spadldf.xG_distance * spadldf.xG_angle)

        # Previous Play Types (Dribble, Throughball, Cutback, Cross, Corner, Freekick)
        spadldf["pass_d"] = np.sqrt(
            (spadldf["start_x"] - spadldf["end_x"]) ** 2
            + (spadldf["start_y"] - spadldf["end_y"]) ** 2
        )
        spadldf["is_forward"] = spadldf.start_x < spadldf.end_x
        spadldf["pass_angle"] = np.degrees(
            np.arctan2(
                spadldf["end_y"] - spadldf["start_y"],
                spadldf["end_x"] - spadldf["start_x"],
            )
        )
        spadldf["is_wide_area"] = (
            (spadldf["start_y"] < 20) | (spadldf["start_y"] > 48)
        ) & (spadldf["start_x"] > 88)

        through_conditions = (
            (spadldf["is_forward"])
            & (spadldf["pass_d"] > 15)
            & (abs(spadldf["pass_angle"]) < 45)
            & (spadldf["type_name"] == "pass")
            & (spadldf["nextEvent"] == "shot")
        )
        cutback_conditions = (
            (~spadldf["is_forward"])
            & (spadldf["is_wide_area"])
            & (4 < spadldf["pass_d"])
            & (spadldf["pass_d"] < 20)
            & (abs(spadldf["pass_angle"]) > 30)
            & (spadldf["type_name"] == "pass")
            & (spadldf["nextEvent"] == "shot")
        )

        spadldf["is_cutback"] = pd.Series(
            np.where(cutback_conditions, True, False)
        ).shift(1)
        spadldf["is_throughball"] = pd.Series(
            np.where(through_conditions, True, False)
        ).shift(1)
        spadldf["after_dribble"] = np.where(
            (spadldf["type_name"] == "shot")
            & (spadldf["prevEvent"].isin(["dribble", "takeon"])),
            True,
            False,
        )

        spadldf["is_corner"] = np.where(
            ((spadldf["type_name"] == "shot") & (spadldf["prevEvent"] == "cross")),
            True,
            False,
        )
        spadldf["is_cross"] = np.where(
            (
                (spadldf["type_name"] == "shot")
                & (spadldf["prevEvent"] == "corner_crossed")
            ),
            True,
            False,
        )
        spadldf["is_freekick"] = np.where(
            (
                (spadldf["type_name"] == "shot")
                & (spadldf["prevEvent"] == "freekick_crossed")
            ),
            True,
            False,
        )

        # if goal was scored and body part used
        spadldf["body"] = np.where(
            (spadldf.bodypart_name.isin(["foot", "foot_right", "foot_left"])),
            "foot",
            "header",
        )
        spadldf["goal"] = np.where(
            (spadldf.type_name == "shot") & (spadldf.result_name == "success"),
            True,
            False,
        )

        # Takes long time (cut down to 12 seconds from 60)
        # spadldf["play_type"] = None

        spadldf = self._calculate_play_type(spadldf)

        self.features_calculated = True
        return spadldf

    def _calculate_relative_angle(self, x, y):
        try:
            # Coordinates of the goalposts
            goal_x = 105
            left_goalpost_y = 30.34
            right_goalpost_y = 37.66
            goal_center_y = 34

            # Validate inputs
            if not (0 <= x <= 105) or not (0 <= y <= 68):
                raise ValueError("Coordinates out of bounds.")

            # Calculate the angles from the player's position to each goalpost
            angle_to_left_post = math.atan2(left_goalpost_y - y, goal_x - x)
            angle_to_right_post = math.atan2(right_goalpost_y - y, goal_x - x)

            # Convert angles to degrees
            angle_to_left_post_deg = math.degrees(angle_to_left_post)
            angle_to_right_post_deg = math.degrees(angle_to_right_post)

            # Calculate the central angle (angle to the goal center)
            central_angle = math.atan2(goal_center_y - y, goal_x - x)
            central_angle_deg = math.degrees(central_angle)

            # Determine the relative angle
            if y > goal_center_y:
                relative_angle = abs(angle_to_left_post_deg - central_angle_deg) / 45
            else:
                relative_angle = abs(angle_to_right_post_deg - central_angle_deg) / 45

            # Ensure the relative angle is between 0 and 1
            relative_angle = max(0, min(1, relative_angle))

            return relative_angle

        except ValueError as e:
            print(f"ValueError: {e}")
            return None

    def _calculate_play_type(self, spadldf):
        spadldf = spadldf.copy()  # To avoid modifying the original dataframe

        # Initialize play_type column
        spadldf["play_type"] = "normal"

        # Group by possession_chain
        grouped = spadldf.groupby("possession_chain")

        # Pre-calculate conditions and masks
        type_name = spadldf["type_name"].values
        start_x = spadldf["start_x"].values
        time_seconds = spadldf["time_seconds"].values

        specific_play_types = [
            "goalkick",
            "freekick_short",
            "freekick_crossed",
            "corner_crossed",
            "corner_short",
        ]

        for name, group in tqdm(grouped, desc="Calculating play types"):
            group_indices = group.index
            first_event_idx = group_indices[0]
            highest_x_event_idx = group["start_x"].idxmax()

            first_event_type_name = type_name[first_event_idx]
            first_event_start_x = start_x[first_event_idx]
            first_event_time_seconds = time_seconds[first_event_idx]

            highest_x_event_time_seconds = time_seconds[highest_x_event_idx]
            highest_x_event_start_x = start_x[highest_x_event_idx]

            if first_event_type_name in specific_play_types:
                play_type = first_event_type_name.split("_")[0]
                spadldf.loc[group_indices, "play_type"] = play_type
            elif (
                (highest_x_event_time_seconds - first_event_time_seconds < 13)
                and (highest_x_event_start_x > 85)
                and (first_event_start_x < 55)
                and (first_event_type_name not in specific_play_types + ["foul"])
            ):
                spadldf.loc[group_indices, "play_type"] = "counter"

        return spadldf
