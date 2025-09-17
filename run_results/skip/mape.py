import polars as pl
import re

def _check_mape(expected, response):
    """Check if entire list contains only numeric values."""
    flt_expected = float(re.findall(r"\d+\.?\d*", expected[-1])[0])
    flt_response = float(re.findall(r"\d+\.?\d*", response[-1])[0])

    mape_val = (abs(flt_expected - flt_response) / flt_expected) * 100

    return True if mape_val <= 10 else False

def eval_dist(df):
    """
    Evaluate distance answers
    """

    df = df.with_columns(
        correct = pl.struct(pl.all()).map_elements(lambda x: _check_mape(x['expected_answer'], x['ovis25_response']))
    )
    return df

pl_data = pl.read_json('/users/2/pyo00005/HOME/carto-reasoning/run_results/skip/ovis25-9B-think_wo_contextual.json').filter(
    pl.col('answer_type') == 'distance'
).with_columns(
    pl.col(['expected_answer', 'ovis25_response']).str.split(';')
)

pl_data = eval_dist(pl_data)

print(pl_data)