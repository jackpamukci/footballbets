import os
import pandas as pd
import boto3
from dotenv import load_dotenv
import socceraction.spadl as spadl


class Events:
    def __init__(
        self,
        league_id: str,
        season_id: int,
        env_path: str = None,
        bucket: str = "footballbets",
    ):

        self.league_id = league_id
        self.season_id = season_id
        self.bucket = bucket
        self.env_path = env_path
        self._get_s3_agent()

        _data = self._load_data()

    def _load_data(self):
        # events = self.s3.get_object(
        #     Bucket=self.bucket,
        #     Key=f"{self.league_id}/{self.season_id}/events_spadl.csv",
        # )
        # spadldf = pd.read_csv(StringIO(spadl_e["Body"].read().decode("utf-8")))
        return

    def _process_event_data(self):
        return

    def _get_s3_agent(self):
        try:
            load_dotenv(self.env_path)
            aws_access_key = os.getenv("AWS_ACCESS_KEY")
            aws_secret_access = os.getenv("AWS_SECRET_ACCESS")
            aws_region = os.getenv("AWS_REGION")

            self.s3 = boto3.client(
                "s3",
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_access,
                region_name=aws_region,
            )
        except Exception as e:
            raise ConnectionError("Connection to AWS Failed. Check Credentials.") from e
