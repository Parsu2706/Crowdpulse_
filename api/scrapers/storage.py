import pandas as pd
from api.config import MAX_STORED_ROWS , ROLLING_WINDOW_DAYS

def merge_and_save(new_rows : list[dict] , path , key: str , date_col : str) -> pd.DataFrame:

    try:
        old = pd.read_csv(path)
    except Exception :
        old = pd.DataFrame()

    combine = pd.concat([old , pd.DataFrame(new_rows)] , ignore_index=True)
    if combine.empty:
        return combine

    combine = combine.drop_duplicates(subset=key , keep='last')

    dt = pd.to_datetime(combine[date_col] , errors="coerce" , utc=True)
    dt = dt.fillna(pd.to_datetime(combine['fetched_at'] , errors="coerce" , utc=True))
    combine["_dt"] = dt 

    cu = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=ROLLING_WINDOW_DAYS)
    combine = combine[combine["_dt"].isna() | (combine["_dt"] >= cu)]
    combine = combine.sort_values("_dt" , ascending=False).head(MAX_STORED_ROWS)

    combine = combine.drop(columns="_dt").reset_index(drop=True)
    combine.to_csv(path , index=False)
    return combine