
import re
from typing import Optional, List, Dict
import numpy as np
import pandas as pd
import streamlit as st
import pydeck as pdk
import altair as alt

st.set_page_config(page_title="FAIL Checker Dashboard", layout="wide", initial_sidebar_state="expanded")

# ---------- Configuration ----------
DATA_PATH_DEFAULT = "FAIL_Checker_Model_Output.xlsx"

CATEGORIES = [
    "Green_Flag",
    "Siyam-siyam_Project",
    "Chop_chop_Project",
    "Doppelganger_Project",
    "Ghost_Project",
]
GREEN_SET = {"Green_Flag"}
RED_SET = {"Siyam-siyam_Project", "Chop_chop_Project", "Doppelganger_Project", "Ghost_Project"}

CAT_COLOR: Dict[str, List[int]] = {
    "Green_Flag": [18, 152, 66, 190],
    "Siyam-siyam_Project": [214, 139, 0, 190],
    "Chop_chop_Project":   [189, 28, 28, 190],
    "Doppelganger_Project":[128, 60, 170, 190],
    "Ghost_Project":       [32, 96, 168, 190],
}
MAP_INITIAL_VIEW = pdk.ViewState(latitude=12.8797, longitude=121.7740, zoom=5, pitch=0)

# ---------- Utilities ----------
@st.cache_data(show_spinner=False)
def load_data(path: str) -> pd.DataFrame:
    return pd.read_excel(path)

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    new_cols = {}
    for c in df.columns:
        nc = re.sub(r"\s+", "_", str(c)).strip("_")
        new_cols[c] = nc
    return df.rename(columns=new_cols)

def find_column(df: pd.DataFrame, candidates):
    cols = list(df.columns)
    # exact
    for c in candidates:
        if c in cols:
            return c
    # case-insensitive exact
    lower = {c.lower(): c for c in cols}
    for c in candidates:
        if c.lower() in lower:
            return lower[c.lower()]
    # fuzzy contains
    for c in cols:
        lc = c.lower()
        if any(k.lower() in lc for k in candidates):
            return c
    return None

def detect_project_id_col(df):  return find_column(df, ["ProjectID","Project_Id","Project_ID","ProjID","ProjectCode","Project_Code"])
def detect_lat_lon_cols(df):    return find_column(df, ["Latitude","Lat"]), find_column(df, ["Longitude","Lon","Lng"])
def detect_deo_col(df):         return find_column(df, ["DistrictEngineeringOffice","District_Engineering_Office","DEO"])
def detect_contractor_col(df):  return find_column(df, ["Contractor","ContractorName","Supplier","SupplierName"])
def detect_year_col(df):        return find_column(df, ["Year","InfraYear","Start_Year","StartYear","CompletionYear","End_Year","EndYear"])
def detect_cost_col(df):
    col = find_column(df, ["ContractCost","Total_Contract_Cost","Contract_Amount","ProjectCost","Approved_Budget_for_the_Contract","ABC"])
    if col: return col
    # keyword fallback
    for c in df.columns:
        lc = c.lower()
        if ("contract" in lc or "project" in lc) and ("cost" in lc or "amount" in lc or "value" in lc):
            return c
    # last resort: first numeric column
    nums = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    return nums[0] if nums else None

def detect_tag_col(df):
    for c in df.columns:
        vals = df[c].dropna().astype(str).head(400).str.lower()
        if vals.empty: 
            continue
        if any(any(cat.lower() in v for v in vals) for cat in CATEGORIES):
            return c
    return find_column(df, ["Tags","Tagging","FAIL_Tags","FAIL_Tag","Category","Categories","Labels"])

def ensure_numeric(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.fillna(0)

def deduplicate_by_last(df: pd.DataFrame, key_col: Optional[str]) -> pd.DataFrame:
    return df.drop_duplicates(subset=[key_col], keep="last") if key_col and key_col in df.columns else df

def explode_tags(df: pd.DataFrame, tag_col: str) -> pd.DataFrame:
    t = df.copy()
    t[tag_col] = t[tag_col].astype(str)
    t["_tags_list"] = t[tag_col].str.split(r"\s*;\s*")
    t = t.explode("_tags_list")
    t["_tags_list"] = t["_tags_list"].str.strip()
    t = t[t["_tags_list"].isin(CATEGORIES)]
    return t.rename(columns={"_tags_list": "Tag"})

def contains_category(tag_string: str, category: str) -> bool:
    return bool(re.search(fr"(^|;)\s*{re.escape(category)}\s*(;|$)", str(tag_string), flags=re.IGNORECASE))

def filter_inclusive(df: pd.DataFrame, tag_col: str, category: str) -> pd.DataFrame:
    return df[df[tag_col].astype(str).apply(lambda s: contains_category(s, category))].copy()

def add_year_if_missing(df: pd.DataFrame, year_col: Optional[str]) -> (pd.DataFrame, Optional[str]):
    if year_col:
        return df, year_col
    # try to derive from any date-like column
    date_candidates = [c for c in df.columns if "date" in c.lower() or "year" in c.lower()]
    for c in date_candidates:
        try:
            d = pd.to_datetime(df[c], errors="coerce")
            if d.notna().sum() > 0:
                df2 = df.copy()
                df2["__Year"] = d.dt.year
                return df2, "__Year"
        except Exception:
            continue
    return df, None

def alt_bar(df: pd.DataFrame, x_col: str, y_col: str, title: str, sort_col: Optional[str] = None, topn: Optional[int] = None):
    d = df.copy()
    if sort_col:
        d = d.sort_values(sort_col, ascending=False)
    if topn:
        d = d.head(topn)
    chart = (
        alt.Chart(d)
        .mark_bar()
        .encode(
            x=alt.X(y_col, title=y_col, type="quantitative"),
            y=alt.Y(x_col, sort="-x", title=x_col),
            tooltip=[x_col, y_col],
        )
        .properties(height=400, title=title)
    )
    st.altair_chart(chart, use_container_width=True)

def green_metrics_per_deo(df: pd.DataFrame, deo_col: str, cost_col: Optional[str], tag_col: str) -> pd.DataFrame:
    w = df.copy()
    w["_is_green"] = w[tag_col].astype(str).apply(lambda s: contains_category(s, "Green_Flag"))
    w["_cost"] = ensure_numeric(w[cost_col]) if cost_col else 0
    total = w.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    green_ct = w.groupby(deo_col, dropna=False)["_is_green"].sum().astype(int).reset_index(name="green_flag_projects")
    if cost_col:
        total_cost = w.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        green_cost = w[w["_is_green"]].groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="green_flag_contract_cost")
    else:
        total_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "total_contract_cost": 0})
        green_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "green_flag_contract_cost": 0})
    out = (
        total.merge(green_ct, on=deo_col, how="left")
             .merge(total_cost, on=deo_col, how="left")
             .merge(green_cost, on=deo_col, how="left")
    ).fillna({"green_flag_projects": 0, "green_flag_contract_cost": 0, "total_contract_cost": 0})
    out["green_flag_projects"] = out["green_flag_projects"].astype(int)
    out["green_flag_density"] = np.where(out["total_projects"] > 0, out["green_flag_projects"]/out["total_projects"], 0.0)
    out["green_flag_cost_ratio"] = np.where(out["total_contract_cost"] > 0, out["green_flag_contract_cost"]/out["total_contract_cost"], 0.0)
    return out.sort_values(["green_flag_projects","green_flag_density"], ascending=[False, False]).reset_index(drop=True)

def category_metrics_per_deo(df: pd.DataFrame, deo_col: str, cost_col: Optional[str], tag_col: str, category: str) -> pd.DataFrame:
    w = df.copy()
    w["_in_cat"] = w[tag_col].astype(str).apply(lambda s: contains_category(s, category))
    w["_cost"] = ensure_numeric(w[cost_col]) if cost_col else 0
    total = w.groupby(deo_col, dropna=False).size().reset_index(name="total_projects")
    cat_ct = w.groupby(deo_col, dropna=False)["_in_cat"].sum().astype(int).reset_index(name="category_projects")
    if cost_col:
        total_cost = w.groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name="total_contract_cost")
        cat_cost = w[w["_in_cat"]].groupby(deo_col, dropna=False)["_cost"].sum().reset_index(name=f"{category}_contract_cost")
    else:
        total_cost = pd.DataFrame({deo_col: w[deo_col].unique(), "total_contract_cost": 0})
        cat_cost = pd.DataFrame({deo_col: w[deo_col].unique(), f"{category}_contract_cost": 0})
    out = (
        total.merge(cat_ct, on=deo_col, how="left")
             .merge(total_cost, on=deo_col, how="left")
             .merge(cat_cost, on=deo_col, how="left")
    ).fillna({"category_projects": 0, f"{category}_contract_cost": 0, "total_contract_cost": 0})
    out["category_projects"] = out["category_projects"].astype(int)
    out["category_density"] = np.where(out["total_projects"] > 0, out["category_projects"]/out["total_projects"], 0.0)
    out["category_cost_ratio"] = np.where(out["total_contract_cost"] > 0, out[f"{category}_contract_cost"]/out["total_contract_cost"], 0.0)
    return out.sort_values(["category_projects","category_density"], ascending=[False, False]).reset_index(drop=True)

def top_entities(df: pd.DataFrame, deo_col: Optional[str], contractor_col: Optional[str], cost_col: Optional[str], by="count", topn=15):
    """
    Robust 'Top' aggregations. For cost mode, convert the cost column once,
    then groupby-sum and sort. Avoid chained apply/sum/sort that can raise AttributeError.
    """
    out = {}
    # DEO
    if deo_col and deo_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = df.copy()
            tmp["_cost"] = ensure_numeric(tmp[cost_col])
            s = tmp.groupby(deo_col, dropna=False)["_cost"].sum().sort_values(ascending=False).head(topn)
            out["DEO"] = s.reset_index(name="Total Cost")
        else:
            s = df.groupby(deo_col, dropna=False).size().sort_values(ascending=False).head(topn)
            out["DEO"] = s.reset_index(name="Projects")
    # Contractor
    if contractor_col and contractor_col in df.columns:
        if by == "cost" and cost_col and cost_col in df.columns:
            tmp = df.copy()
            tmp["_cost"] = ensure_numeric(tmp[cost_col])
            s = tmp.groupby(contractor_col, dropna=False)["_cost"].sum().sort_values(ascending=False).head(topn)
            out["Contractor"] = s.reset_index(name="Total Cost")
        else:
            s = df.groupby(contractor_col, dropna=False).size().sort_values(ascending=False).head(topn)
            out["Contractor"] = s.reset_index(name="Projects")
    return out

def map_layers_for_categories(df: pd.DataFrame, lat_col: str, lon_col: str, tag_col: str, cats: List[str]):
    layers = []
    for cat in cats:
        dd = df[df[tag_col].astype(str).apply(lambda s: contains_category(s, cat))]
        dd = dd.dropna(subset=[lat_col, lon_col])
        if dd.empty:
            continue
        layers.append(
            pdk.Layer(
                "ScatterplotLayer",
                data=dd,
                get_position=[lon_col, lat_col],
                get_radius=80,
                pickable=True,
                radius_min_pixels=3,
                radius_max_pixels=8,
                get_fill_color=CAT_COLOR.get(cat, [120,120,120,160]),
                auto_highlight=True,
            )
        )
    return layers

def deck_chart(layers, tooltip=None):
    # No map_style arg to avoid AttributeError in some environments
    return pdk.Deck(initial_view_state=MAP_INITIAL_VIEW, layers=layers, tooltip=tooltip)

# ---------- Load and prepare ----------
try:
    df_raw = load_data(DATA_PATH_DEFAULT)
except Exception as e:
    st.error(f"Failed to read Excel file '{DATA_PATH_DEFAULT}'. Error: {e}")
    st.stop()

df = normalize_columns(df_raw)

# Auto-detect columns (no UI)
proj_col = detect_project_id_col(df)
lat_col, lon_col = detect_lat_lon_cols(df)
deo_col = detect_deo_col(df)
contractor_col = detect_contractor_col(df)
cost_col = detect_cost_col(df)
tag_col = detect_tag_col(df)
year_col = detect_year_col(df)

# Validate essential columns
missing = []
if tag_col is None: missing.append("Tag/Category column (must contain labels like 'Green_Flag', etc.)")
if lat_col is None or lon_col is None: missing.append("Latitude/Longitude")
if deo_col is None: missing.append("DistrictEngineeringOffice")
if missing:
    st.error("Missing required columns: " + "; ".join(missing))
    st.stop()

# Deduplicate by ProjectID
df = deduplicate_by_last(df, proj_col)

# Add year if needed
df, derived_year_col = add_year_if_missing(df, year_col)
year_col = year_col or derived_year_col

# Working dataframe (you can add global filters here later if desired)
working = df.copy()

# ---------- Tabs ----------
tabs = st.tabs(["Overview", "Compare", "Green Flag", "Siyam-siyam", "Chop-chop", "Doppelganger", "Ghost"])

# ===== Overview =====
with tabs[0]:
    st.header("Overview")

    total_projects = len(working)
    exploded = explode_tags(working, tag_col)
    tag_counts = exploded["Tag"].value_counts().reindex(CATEGORIES, fill_value=0)
    green_count = int(tag_counts.get("Green_Flag", 0))
    red_count = int(tag_counts.reindex(list(RED_SET), fill_value=0).sum())
    green_share = (green_count / total_projects * 100) if total_projects > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Projects", f"{total_projects:,}")
    c2.metric("Green Flag Projects", f"{green_count:,}")
    c3.metric("Red-tagged Projects", f"{red_count:,}")
    c4.metric("Green Share (%)", f"{green_share:.2f}")

    dist_df = tag_counts.rename_axis("Tag").reset_index(name="Projects")
    alt_bar(dist_df, x_col="Tag", y_col="Projects", title="Distribution of Tags", sort_col="Projects")

    st.subheader("Comparison Map — Green vs Red")
    d = working.copy()
    d["_is_green"] = d[tag_col].astype(str).apply(lambda s: contains_category(s, "Green_Flag"))
    d["_is_red"]   = d[tag_col].astype(str).apply(lambda s: any(contains_category(s, c) for c in RED_SET))
    greens = d[d["_is_green"] & d[lat_col].notna() & d[lon_col].notna()]
    reds   = d[d["_is_red"]   & d[lat_col].notna() & d[lon_col].notna()]
    deck = deck_chart(
        layers=[
            pdk.Layer("ScatterplotLayer", data=greens, get_position=[lon_col, lat_col],
                      get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                      get_fill_color=CAT_COLOR["Green_Flag"], auto_highlight=True),
            pdk.Layer("ScatterplotLayer", data=reds, get_position=[lon_col, lat_col],
                      get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                      get_fill_color=[200, 40, 40, 190], auto_highlight=True),
        ],
        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
    )
    st.pydeck_chart(deck)

    st.subheader("Top Entities")
    mode = st.radio("Rank by", ["Project Count", "Total Cost"], horizontal=True, index=0)
    mode_key = "count" if mode == "Project Count" else "cost"
    tops = top_entities(working, deo_col, contractor_col, cost_col, by=mode_key, topn=15)
    t1, t2 = st.columns(2)
    if "DEO" in tops: t1.dataframe(tops["DEO"])
    if "Contractor" in tops: t2.dataframe(tops["Contractor"])

# ===== Compare =====
with tabs[1]:
    st.header("Category Comparison Map")
    layers = map_layers_for_categories(working, lat_col, lon_col, tag_col, CATEGORIES)
    deck = deck_chart(layers=layers, tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"})
    st.pydeck_chart(deck)

    st.subheader("Category Counts")
    dist_df = explode_tags(working, tag_col)["Tag"].value_counts().reindex(CATEGORIES, fill_value=0).rename_axis("Tag").reset_index(name="Projects")
    st.dataframe(dist_df)
    alt_bar(dist_df, x_col="Tag", y_col="Projects", title="Counts by Category", sort_col="Projects")

# ===== Green Flag =====
with tabs[2]:
    st.header("Green Flag Analysis")
    tmp = working.rename(columns={deo_col: "DistrictEngineeringOffice", (cost_col or "ContractCost"): "ContractCost", tag_col: "Tagging"})
    gm = green_metrics_per_deo(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging")
    st.subheader("Per-DEO Metrics")
    st.dataframe(gm[[
        "DistrictEngineeringOffice",
        "total_projects",
        "green_flag_projects",
        "total_contract_cost",
        "green_flag_contract_cost",
        "green_flag_density",
        "green_flag_cost_ratio"
    ]])
    alt_bar(gm.rename(columns={"DistrictEngineeringOffice":"DEO","green_flag_projects":"Projects"}),
            x_col="DEO", y_col="Projects", title="Green Flag Projects by DEO", sort_col="Projects", topn=25)

    st.subheader("Map")
    gdf = filter_inclusive(working.assign(Tagging=working[tag_col]), "Tagging", "Green_Flag")
    deck = deck_chart(
        layers=[pdk.Layer("ScatterplotLayer", data=gdf.dropna(subset=[lat_col,lon_col]),
                          get_position=[lon_col,lat_col], get_radius=80, pickable=True,
                          radius_min_pixels=3, radius_max_pixels=8, get_fill_color=CAT_COLOR["Green_Flag"], auto_highlight=True)],
        tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
    )
    st.pydeck_chart(deck)

    st.subheader("Table")
    st.dataframe(filter_inclusive(working, tag_col, "Green_Flag"))

# Helper to render a category tab with per-DEO metrics + map + table
def render_category_tab(df_in: pd.DataFrame, category: str, container):
    with container:
        st.header(category.replace("_"," "))
        tmp = df_in.rename(columns={deo_col: "DistrictEngineeringOffice", (cost_col or "ContractCost"): "ContractCost", tag_col: "Tagging"})
        cm = category_metrics_per_deo(tmp, "DistrictEngineeringOffice", "ContractCost", "Tagging", category)
        st.subheader("Per-DEO Metrics")
        cols = ["DistrictEngineeringOffice","total_projects","category_projects","total_contract_cost",f"{category}_contract_cost","category_density","category_cost_ratio"]
        st.dataframe(cm[cols])
        alt_bar(cm.rename(columns={"DistrictEngineeringOffice":"DEO","category_projects":"Projects"}),
                x_col="DEO", y_col="Projects", title=f"{category} Projects by DEO", sort_col="Projects", topn=25)

        st.subheader("Map")
        cdf = filter_inclusive(df_in.assign(Tagging=df_in[tag_col]), "Tagging", category)
        deck = deck_chart(
            layers=[pdk.Layer("ScatterplotLayer", data=cdf.dropna(subset=[lat_col,lon_col]),
                              get_position=[lon_col,lat_col],
                              get_radius=80, pickable=True, radius_min_pixels=3, radius_max_pixels=8,
                              get_fill_color=CAT_COLOR.get(category,[120,120,120,160]), auto_highlight=True)],
            tooltip={"text": "{ProjectID}\n{DistrictEngineeringOffice}\n{Contractor}\n{Tagging}"}
        )
        st.pydeck_chart(deck)

        st.subheader("Table")
        st.dataframe(filter_inclusive(df_in, tag_col, category))

# ===== Siyam-siyam =====
render_category_tab(working, "Siyam-siyam_Project", tabs[3])

# ===== Chop-chop =====
render_category_tab(working, "Chop_chop_Project", tabs[4])

# ===== Doppelganger =====
render_category_tab(working, "Doppelganger_Project", tabs[5])

# ===== Ghost =====
render_category_tab(working, "Ghost_Project", tabs[6])
