from __future__ import annotations

from datetime import date, datetime, time
from difflib import SequenceMatcher
from io import BytesIO
from typing import Any

import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import Date, DateTime, Integer, LargeBinary, String, Text, create_engine, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker
from sqlalchemy.pool import StaticPool


st.set_page_config(
    page_title="AI Lost & Found Matcher",
    page_icon="🔎",
    layout="wide",
    initial_sidebar_state="expanded",
)


class Base(DeclarativeBase):
    pass


class ItemReport(Base):
    __tablename__ = "item_reports"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    kind: Mapped[str] = mapped_column(String(10), nullable=False, index=True)
    item_name: Mapped[str] = mapped_column(String(160), nullable=False)
    category: Mapped[str] = mapped_column(String(80), nullable=False)
    color: Mapped[str] = mapped_column(String(80), nullable=False)
    brand: Mapped[str] = mapped_column(String(120), nullable=False, default="")
    description: Mapped[str] = mapped_column(Text, nullable=False)
    location: Mapped[str] = mapped_column(String(200), nullable=False)
    item_date: Mapped[date] = mapped_column(Date, nullable=False)
    approx_time: Mapped[str] = mapped_column(String(20), nullable=False)
    image_data: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    contact: Mapped[str] = mapped_column(String(200), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=datetime.utcnow)


@st.cache_resource
def get_database() -> sessionmaker[Session]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, expire_on_commit=False)


@st.cache_resource(show_spinner=False)
def load_text_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def load_clip_model() -> Any:
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("clip-ViT-B-32")


def get_reports(kind: str | None = None) -> list[ItemReport]:
    with get_database()() as db:
        query = select(ItemReport).order_by(ItemReport.created_at.desc())
        if kind:
            query = query.where(ItemReport.kind == kind)
        return list(db.scalars(query).all())


def make_report_text(report: ItemReport) -> str:
    return ". ".join(
        part
        for part in (
            report.item_name,
            report.category,
            report.color,
            report.brand,
            report.description,
            report.location,
        )
        if part and part.strip()
    )


def normalized_similarity(first: str, second: str) -> float:
    first_clean = " ".join(first.casefold().split())
    second_clean = " ".join(second.casefold().split())
    if not first_clean or not second_clean:
        return 0.0
    if first_clean == second_clean:
        return 1.0
    return SequenceMatcher(None, first_clean, second_clean).ratio()


def token_similarity(first: str, second: str) -> float:
    first_words = {word.casefold().strip(".,!?;:()[]{}") for word in first.split()}
    second_words = {word.casefold().strip(".,!?;:()[]{}") for word in second.split()}
    first_words.discard("")
    second_words.discard("")
    if not first_words or not second_words:
        return 0.0
    overlap = len(first_words & second_words) / len(first_words | second_words)
    return max(overlap, normalized_similarity(first, second) * 0.7)


def calculate_date_similarity(lost: ItemReport, found: ItemReport) -> float:
    day_difference = abs((lost.item_date - found.item_date).days)
    day_score = max(0.0, 1.0 - day_difference / 31.0)
    if not lost.approx_time or not found.approx_time:
        return day_score

    try:
        lost_time = time.fromisoformat(lost.approx_time)
        found_time = time.fromisoformat(found.approx_time)
        lost_minutes = lost_time.hour * 60 + lost_time.minute
        found_minutes = found_time.hour * 60 + found_time.minute
        time_difference = abs(lost_minutes - found_minutes)
        time_difference = min(time_difference, 24 * 60 - time_difference)
        time_score = max(0.0, 1.0 - time_difference / (12 * 60))
        return day_score * 0.75 + time_score * 0.25
    except ValueError:
        return day_score


def image_from_bytes(image_data: bytes) -> Image.Image:
    with Image.open(BytesIO(image_data)) as image:
        image.load()
        return image.convert("RGB")


def build_match_scores(
    lost: ItemReport,
    found_reports: list[ItemReport],
) -> tuple[list[dict[str, Any]], str | None, str | None]:
    text_warning: str | None = None
    image_warning: str | None = None
    text_scores: dict[int, float] = {}
    image_scores: dict[int, float] = {}

    try:
        text_model = load_text_model()
        texts = [make_report_text(lost), *(make_report_text(report) for report in found_reports)]
        embeddings = text_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        similarities = cosine_similarity(embeddings[:1], embeddings[1:])[0]
        text_scores = {
            report.id: float(np.clip(score, 0.0, 1.0))
            for report, score in zip(found_reports, similarities)
        }
    except Exception as exc:
        text_warning = (
            "The sentence-embedding model could not be loaded. Text scores are using a "
            f"simple word-overlap fallback for this search ({type(exc).__name__})."
        )
        text_scores = {
            report.id: token_similarity(make_report_text(lost), make_report_text(report))
            for report in found_reports
        }

    image_pairs = [
        report
        for report in found_reports
        if lost.image_data is not None and report.image_data is not None
    ]
    if image_pairs:
        try:
            clip_model = load_clip_model()
            lost_image = image_from_bytes(lost.image_data)
            images = [lost_image, *(image_from_bytes(report.image_data) for report in image_pairs)]
            image_embeddings = clip_model.encode(images, convert_to_numpy=True, normalize_embeddings=True)
            image_similarities = cosine_similarity(image_embeddings[:1], image_embeddings[1:])[0]
            image_scores = {
                report.id: float(np.clip(score, 0.0, 1.0))
                for report, score in zip(image_pairs, image_similarities)
            }
        except Exception as exc:
            image_warning = (
                "CLIP image matching is unavailable for this search; the overall score "
                f"has been recalculated without image similarity ({type(exc).__name__})."
            )

    results: list[dict[str, Any]] = []
    for found in found_reports:
        factors = {
            "Text": text_scores.get(found.id, 0.0),
            "Category": normalized_similarity(lost.category, found.category),
            "Color": token_similarity(lost.color, found.color),
            "Location": token_similarity(lost.location, found.location),
            "Date": calculate_date_similarity(lost, found),
        }
        weights = {"Text": 0.40, "Image": 0.30, "Category": 0.10, "Color": 0.10, "Location": 0.05, "Date": 0.05}
        if found.id in image_scores:
            factors["Image"] = image_scores[found.id]
        else:
            weights.pop("Image")

        active_weight = sum(weights[name] for name in factors)
        overall = sum(factors[name] * weights[name] for name in factors) / active_weight
        results.append({"report": found, "scores": factors, "overall": overall})

    results.sort(key=lambda result: result["overall"], reverse=True)
    return results, text_warning, image_warning


def match_band(score: float) -> tuple[str, str]:
    if score >= 0.80:
        return "High Match", "🟢"
    if score >= 0.60:
        return "Medium Match", "🟡"
    return "Low Match", "⚪"


def save_report(
    kind: str,
    item_name: str,
    category: str,
    color: str,
    brand: str,
    description: str,
    location: str,
    item_date: date,
    approx_time: str,
    image_data: bytes | None,
    contact: str,
) -> None:
    report = ItemReport(
        kind=kind,
        item_name=item_name.strip(),
        category=category.strip(),
        color=color.strip(),
        brand=brand.strip(),
        description=description.strip(),
        location=location.strip(),
        item_date=item_date,
        approx_time=approx_time,
        image_data=image_data,
        contact=contact.strip(),
    )
    with get_database()() as db:
        db.add(report)
        db.commit()


def render_report_form(kind: str) -> None:
    is_lost = kind == "lost"
    heading = "Report a Lost Item" if is_lost else "Report a Found Item"
    st.title(heading)
    st.caption(
        "Share clear details so the matcher can compare it with " +
        ("found reports." if is_lost else "lost reports.")
    )

    with st.form(f"{kind}_report_form", clear_on_submit=True):
        left, right = st.columns(2)
        with left:
            item_name = st.text_input("Item name *", placeholder="Item name (e.g., Black leather wallet)")
            category = st.selectbox(
                "Category",
                ["Wallet", "Phone", "Keys", "Bag", "Clothing", "Electronics", "Jewelry", "ID / Documents", "Other"],
            )
            color = st.text_input("Color", placeholder="Color (e.g., black)")
            brand = st.text_input("Brand (optional)", placeholder="Brand (e.g., Fossil)")
        with right:
            location = st.text_input("Location *", placeholder="Location (e.g., Central Library, 2nd Floor)")
            item_date = st.date_input("Date *", value=date.today())
            approx_time = st.time_input("Approximate time", value=time(12, 0)).strftime("%H:%M")
            contact = st.text_input("Contact information (optional)", placeholder="Email or phone")
        description = st.text_area(
            "Description *",
            placeholder="Describe distinctive features, contents, marks, or other useful details.",
            height=120,
            key=f"{kind}_description",
        )
        uploaded_image = st.file_uploader(
            "Image (optional)",
            type=["jpg", "jpeg", "png", "webp"],
            help="Images are held in the app's in-memory database and used for visual matching.",
        )
        submitted = st.form_submit_button("Save report", type="primary", use_container_width=True)

    if submitted:
        if not item_name.strip() or not location.strip() or not description.strip():
            st.error("Please complete the item name, location, and description before saving.")
            return

        image_data: bytes | None = None
        if uploaded_image is not None:
            try:
                image_data = uploaded_image.getvalue()
                image_from_bytes(image_data)
            except (UnidentifiedImageError, OSError, ValueError):
                st.error("That image could not be read. Please upload a valid JPG, PNG, or WEBP image.")
                return

        try:
            save_report(
                kind, item_name, category, color, brand, description,
                location, item_date, approx_time, image_data, contact,
            )
        except Exception as exc:
            st.error(f"We couldn't save this report ({type(exc).__name__}). Please try again.")
            return
        st.success(f"{'Lost' if is_lost else 'Found'} item report saved successfully.")


def render_item_details(report: ItemReport, scores: dict[str, float] | None = None, overall: float | None = None) -> None:
    if report.image_data:
        try:
            st.image(image_from_bytes(report.image_data), use_container_width=True)
        except (UnidentifiedImageError, OSError, ValueError):
            st.info("The saved image is no longer readable.")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**Category:** {report.category}")
        st.markdown(f"**Color:** {report.color or 'Not specified'}")
        st.markdown(f"**Brand:** {report.brand or 'Not specified'}")
        st.markdown(f"**Location:** {report.location}")
    with right:
        st.markdown(f"**Date:** {report.item_date.strftime('%B %d, %Y')}")
        st.markdown(f"**Approximate time:** {report.approx_time or 'Not specified'}")
        st.markdown(f"**Contact:** {report.contact or 'Not provided'}")
        if overall is not None:
            label, icon = match_band(overall)
            st.markdown(f"**Overall match:** {icon} {overall:.0%} — {label}")
    st.markdown("**Description**")
    st.write(report.description or "No description provided.")
    if scores:
        st.markdown("**Similarity breakdown**")
        score_columns = st.columns(len(scores))
        for column, (name, score) in zip(score_columns, scores.items()):
            column.metric(f"{name} Match", f"{score:.0%}")


def render_find_matches() -> None:
    st.title("Find AI Matches")
    st.caption("Choose one of your lost reports to compare it with all available found reports.")
    lost_reports = get_reports("lost")
    found_reports = get_reports("found")
    if not lost_reports:
        st.info("There are no lost-item reports yet. Add one using **Report Lost Item**.")
        return
    if not found_reports:
        st.info("There are no found-item reports yet. Add one using **Report Found Item**.")
        return

    report_names = {
        report.id: f"{report.item_name} · {report.category} · {report.item_date:%b %d, %Y} (#{report.id})"
        for report in lost_reports
    }
    lost_id = st.selectbox("Select your lost item", list(report_names), format_func=report_names.get)
    lost = next(report for report in lost_reports if report.id == lost_id)
    if st.button("Find best matches", type="primary", use_container_width=False):
        with st.spinner("Comparing report details and available images..."):
            results, text_warning, image_warning = build_match_scores(lost, found_reports)
        st.session_state["match_results"] = results
        st.session_state["match_lost_id"] = lost.id
        st.session_state["match_warnings"] = [warning for warning in (text_warning, image_warning) if warning]

    if st.session_state.get("match_lost_id") != lost.id:
        st.info("Select **Find best matches** to run the AI comparison.")
        return

    for warning in st.session_state.get("match_warnings", []):
        st.warning(warning)
    results = st.session_state.get("match_results", [])
    if not results:
        st.info("No found-item reports are available to compare.")
        return

    categories = sorted({entry["report"].category for entry in results})
    colors = sorted({entry["report"].color for entry in results if entry["report"].color})
    locations = sorted({entry["report"].location for entry in results})
    with st.expander("Filter and sort results", expanded=True):
        f1, f2, f3 = st.columns(3)
        category_filter = f1.selectbox("Category", ["All"] + categories)
        color_filter = f2.selectbox("Color", ["All"] + colors)
        location_filter = f3.selectbox("Location", ["All"] + locations)
        minimum_score = st.slider("Minimum match score", 0, 100, 0, 5)
        date_range = st.date_input(
            "Found date range",
            value=(min(entry["report"].item_date for entry in results), max(entry["report"].item_date for entry in results)),
        )

    filtered = results
    if category_filter != "All":
        filtered = [entry for entry in filtered if entry["report"].category == category_filter]
    if color_filter != "All":
        filtered = [entry for entry in filtered if entry["report"].color == color_filter]
    if location_filter != "All":
        filtered = [entry for entry in filtered if entry["report"].location == location_filter]
    if isinstance(date_range, tuple) and len(date_range) == 2:
        filtered = [entry for entry in filtered if date_range[0] <= entry["report"].item_date <= date_range[1]]
    filtered = [entry for entry in filtered if entry["overall"] * 100 >= minimum_score]
    filtered.sort(key=lambda entry: entry["overall"], reverse=True)

    if not filtered:
        st.info("No reports meet those filters. Try widening your search.")
        return

    st.markdown(f"### {len(filtered)} possible match{'es' if len(filtered) != 1 else ''}")
    for entry in filtered:
        found = entry["report"]
        overall = entry["overall"]
        label, icon = match_band(overall)
        with st.container(border=True):
            image_column, info_column = st.columns([1, 2.5])
            with image_column:
                if found.image_data:
                    try:
                        st.image(image_from_bytes(found.image_data), use_container_width=True)
                    except (UnidentifiedImageError, OSError, ValueError):
                        st.caption("Image unavailable")
                else:
                    st.markdown(
                        '<div class="image-placeholder">📦<br/>No image</div>',
                        unsafe_allow_html=True,
                    )
            with info_column:
                st.caption(f"POSSIBLE MATCH · {icon} {label.upper()}")
                st.subheader(found.item_name)
                st.markdown(f"### Match Score: **{overall:.0%}**")
                st.caption(
                    f"{found.category} · {found.color or 'Color not specified'} · "
                    f"{found.location} · {found.item_date:%b %d, %Y}"
                )
                score_cols = st.columns(len(entry["scores"]))
                for column, (name, score) in zip(score_cols, entry["scores"].items()):
                    column.metric(f"{name} Match", f"{score:.0%}")
                signals = [
                    ("Similar description", entry["scores"]["Text"] >= 0.55),
                    ("Same category", entry["scores"]["Category"] >= 0.80),
                    ("Similar color", entry["scores"]["Color"] >= 0.50),
                    ("Nearby location", entry["scores"]["Location"] >= 0.40),
                    ("Similar date", entry["scores"]["Date"] >= 0.60),
                ]
                st.caption("   ·   ".join(f"{'✓' if matched else '○'} {label}" for label, matched in signals))
                with st.expander(f"View details — {found.item_name}"):
                    render_item_details(found, entry["scores"], overall)


def navigate_to(page: str) -> None:
    navigation_labels = {
        "home": "Home",
        "lost": "Report Lost Item",
        "found": "Report Found Item",
        "matches": "Find AI Matches",
    }
    st.session_state["page"] = page
    st.session_state["navigation"] = navigation_labels[page]


def render_home() -> None:
    st.markdown('<div class="hero-kicker">CAMPUS COMMUNITY · SMARTER RECOVERY</div>', unsafe_allow_html=True)
    st.title("AI Lost & Found Matcher")
    st.markdown('<p class="tagline">Find what you lost. Match what you found.</p>', unsafe_allow_html=True)
    st.write(
        "Report an item or search existing reports. Local AI compares descriptions and, "
        "when both reports include images, visual features to rank likely matches."
    )

    pages = [
        ("Report Lost Item", "Tell the community what you're looking for.", "lost"),
        ("Report Found Item", "Help reunite an item with its owner.", "found"),
        ("Find AI Matches", "Compare lost reports with possible finds.", "matches"),
    ]
    columns = st.columns(3)
    for column, (label, description, page) in zip(columns, pages):
        with column:
            with st.container(border=True):
                st.markdown(f"#### {label}")
                st.caption(description)
                st.button(
                    "Open",
                    key=f"home_{page}",
                    use_container_width=True,
                    on_click=navigate_to,
                    args=(page,),
                )

    st.markdown("### How matching works")
    steps = [
        ("01", "Report item"),
        ("02", "AI understands details"),
        ("03", "Compare lost & found"),
        ("04", "Calculate similarity"),
        ("05", "Show best matches"),
    ]
    flow = st.columns(len(steps))
    for index, (column, (number, label)) in enumerate(zip(flow, steps)):
        with column:
            st.markdown(f'<div class="flow-step"><span>{number}</span><br/><b>{label}</b></div>', unsafe_allow_html=True)
        if index < len(steps) - 1:
            flow[index].markdown('<div class="flow-arrow">↓</div>', unsafe_allow_html=True)

    lost_count = len(get_reports("lost"))
    found_count = len(get_reports("found"))
    st.markdown("---")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Lost reports", lost_count)
    metric_columns[1].metric("Found reports", found_count)
    metric_columns[2].metric("AI match engine", "Ready")


def main() -> None:
    st.markdown(
        """
        <style>
        .stApp { background: #f5f7fb; color: #172033; }
        [data-testid="stSidebar"] { background: #111827; }
        [data-testid="stSidebar"] * { color: #edf2ff; }
        [data-testid="stTextArea"] textarea {
            color: #172033 !important;
            background-color: #ffffff !important;
            caret-color: #4f46e5 !important;
        }
        [data-testid="stTextArea"] textarea::placeholder { color: #68748b !important; opacity: 1; }
        .hero-kicker { color: #4f46e5; font-size: .78rem; font-weight: 800; letter-spacing: .12em; }
        .tagline { color: #4b5568; font-size: 1.35rem; margin-top: -.7rem; }
        .flow-step { min-height: 92px; padding: 14px; background: white; border: 1px solid #e5e9f2; border-radius: 12px; text-align: center; }
        .flow-step span { color: #6366f1; font-size: .76rem; font-weight: 800; }
        .flow-step b { font-size: .9rem; }
        .flow-arrow { color: #6366f1; text-align: center; font-size: 1.4rem; margin-top: -66px; margin-left: 90%; position: relative; }
        .image-placeholder { display: grid; place-content: center; min-height: 145px; background: #eef1f8; border-radius: 12px; color: #68748b; text-align: center; }
        div[data-testid="stMetric"] { background: #f8f9fd; padding: 10px 12px; border-radius: 10px; }
        </style>
        """,
        unsafe_allow_html=True,
    )
    if "page" not in st.session_state:
        st.session_state["page"] = "home"
    navigation_labels = {
        "home": "Home",
        "lost": "Report Lost Item",
        "found": "Report Found Item",
        "matches": "Find AI Matches",
    }
    if "navigation" not in st.session_state:
        st.session_state["navigation"] = navigation_labels[st.session_state["page"]]

    with st.sidebar:
        st.markdown("## 🔎 AI Matcher")
        st.caption("A smarter way to reunite people with their belongings.")
        navigation_options = list(navigation_labels.values())
        page = st.radio(
            "Navigate",
            navigation_options,
            key="navigation",
            label_visibility="collapsed",
        )
        selected_page = {label: key for key, label in navigation_labels.items()}[page]
        st.session_state["page"] = selected_page
        st.markdown("---")
        st.caption("Demo reports are stored in memory and reset when the app stops.")

    page = st.session_state["page"]
    if page == "lost":
        render_report_form("lost")
    elif page == "found":
        render_report_form("found")
    elif page == "matches":
        render_find_matches()
    else:
        render_home()


if __name__ == "__main__":
    main()
