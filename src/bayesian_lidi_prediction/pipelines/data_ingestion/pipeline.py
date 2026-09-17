from kedro.pipeline import Node, Pipeline  # noqa
from.nodes import convert_csv_to_parquet


def create_pipeline(**kwargs) -> Pipeline:
    return Pipeline([
        Node(
            func=convert_csv_to_parquet,
            inputs="params:train_data_file_path",
            outputs="train_df"
        ),
        Node(
            func=convert_csv_to_parquet,
            inputs="params:test_data_file_path",
            outputs="test_df"
        )
    ])
