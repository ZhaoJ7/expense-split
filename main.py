import io

import streamlit as st
import pandas as pd

from config import DATA_DIR
from pathlib import Path

from src import solver


@st.cache_data
def _load_template_as_io_buffer(fp: str | Path = DATA_DIR / "template.xlsx"):
    buffer = io.BytesIO()
    pd.read_excel(fp).to_excel(excel_writer=buffer)  # noqa
    return buffer


def _process_upload(file: io.BytesIO) -> pd.DataFrame:
    df = pd.read_excel(file)

    # TODO: use some better validation, such as pandera
    assert {"Amount", "Who paid"}.issubset(set(df.columns))
    assert not df["Amount"].isna().any()
    assert not df["Who paid"].isna().any()

    df = df.fillna(0.0)
    return df


def _run_solver(data: pd.DataFrame) -> solver.SolverResult:

    solver_ = solver.Solver(data=data)
    solver_result = solver_.run()
    st.session_state.solver_has_run = True

    return solver_result


def _display_solver_results(solver_res: solver.SolverResult) -> None:

    st.markdown("#### Transfers")
    for transfer in solver_res.transfers:
        st.markdown(
            f"{transfer.from_} transfers ${transfer.amount: .2f} to {transfer.to}"
        )

    st.markdown("#### Transfer summary")
    st.markdown(f"Total number of transfers: {solver_res.total_num_transfers: .0f}")
    st.markdown(f"Total amount transferred: ${solver_res.total_transfer_amount: .2f}")


def main_page():
    st.set_page_config(page_title="Expense Splitting Tool", layout="centered")

    if "solver_has_run" not in st.session_state:
        st.session_state.solver_has_run = False

    template: io.BytesIO = _load_template_as_io_buffer()

    st.markdown("# Expense Splitting Tool")

    # ----- Section: Input data ----- #
    st.markdown("## Input data")
    with st.expander("Input data", expanded=True):
        st.markdown("### Step 1: Download an Excel template")

        c1, _ = st.columns([0.5, 0.5])
        with c1:
            st.download_button(
                "Download template",
                data=template,
                file_name="expense-splitting-input.xlsx",
                mime="text/xlsx",
                use_container_width=True,
                key="download_button",
            )

        st.markdown("### Step 2: Complete the Excel template")

        st.markdown("### Step 3: Upload the completed template")

        def _reset_has_run():
            st.session_state.solver_has_run = False

        uploaded_file = st.file_uploader(
            "Upload expense splitting input",
            type=["xlsx"],
            key=f"file_uploader",
            on_change=_reset_has_run,
        )  # noqa

        if uploaded_file is not None:
            uploaded_df = _process_upload(uploaded_file)
            st.info("Here is your uploaded data. You can make edits if needed:")
            uploaded_df = st.data_editor(uploaded_df, num_rows="dynamic")
        else:
            uploaded_df = None
            st.session_state.solver_has_run = False

        st.markdown("### Step 4: Run the solver")
        c1, _ = st.columns([0.5, 0.5])
        with c1:
            button_run_solver = st.button(
                "Run solver",
                key="button_run_solver",
                disabled=uploaded_df is None,
                use_container_width=True,
                type="primary",
            )
            if button_run_solver:
                solver_res = _run_solver(uploaded_df)
            else:
                solver_res = None

    # ----- Section: Results ----- #
    st.markdown("## Results")
    with st.expander("Results"):
        if st.session_state.solver_has_run:
            st.markdown("### Solver results")
            _display_solver_results(solver_res)

        else:
            st.info("Run the solver to see the results")


if __name__ == "__main__":
    main_page()
