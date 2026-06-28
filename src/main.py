"""
main.py — Streamlit UI for ChessVision.

Run with:
    streamlit run main.py
"""

import tempfile
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

import config
import chess_vision as cv_pipeline
from utils import board_to_fen, board_to_text


# ── Page setup ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="ChessVision", layout="wide")
st.title("♟ ChessVision")

left, mid, right = st.columns([1, 2, 1])

# ── Sidebar controls ──────────────────────────────────────────────────────────
with left:
    st.header("Settings")
    mode = st.radio("Input source", ["Image", "Video", "Webcam (live)", "Webcam snapshot"])
    config.CONFIDENCE = st.slider("Confidence", 0.10, 0.90, config.CONFIDENCE, 0.05)
    st.caption(f"Model: `{Path(config.MODEL_PATH).name}`")


# ── Helper: run the full pipeline on one frame ────────────────────────────────
def process(frame: np.ndarray) -> dict:
    quad        = cv_pipeline.detect_board(frame)
    board_img   = cv_pipeline.rectify_board(frame, quad)
    # Only pass board_img for board-space detection when a real board quad was found.
    # When quad is None, board_img is just a resize of the full frame — running
    # detection on it again would give inconsistent labels for the same pieces.
    detections  = cv_pipeline.detect_pieces(frame, board_img if quad is not None else None)
    board_state = cv_pipeline.create_board_state(detections, frame, quad)
    ann_frame, ann_board = cv_pipeline.draw_results(frame, quad, board_img, detections)
    return {
        "ann_frame":  ann_frame,
        "ann_board":  ann_board,
        "board":      board_state,
        "detections": detections,
        "quad":       quad,
    }


# ── Helper: show board state in right panel ───────────────────────────────────
def show_results(result: dict):
    with right:
        n = len(result["detections"])
        board_status = "Board detected" if result["quad"] is not None else "Board not found"
        st.caption(f"{board_status} · {n} piece{'s' if n != 1 else ''} detected")
        st.subheader("Board state")
        st.code(board_to_text(result["board"]))
        st.subheader("FEN")
        st.code(board_to_fen(result["board"]))

def _write_board_results(result: dict, container):
    """Write board state into a replaceable container (used during video scan)."""
    n = len(result["detections"])
    board_status = "Board detected" if result["quad"] is not None else "Board not found"
    with container.container():
        st.caption(f"{board_status} \u00b7 {n} piece{'s' if n != 1 else ''} detected")
        st.subheader("Board state")
        st.code(board_to_text(result["board"]))
        st.subheader("FEN")
        st.code(board_to_fen(result["board"]))

# ── Image mode ────────────────────────────────────────────────────────────────
if mode == "Image":
    uploaded = st.file_uploader(
        "Upload a chess image",
        type=["jpg", "jpeg", "png", "bmp", "webp"],
        key="img",
    )
    if uploaded:
        frame  = cv_pipeline.load_input(uploaded.read())
        result = process(frame)
        with mid:
            st.subheader("Detected pieces")
            st.image(cv2.cvtColor(result["ann_frame"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            if result["quad"] is not None:
                with st.expander("Corrected board view"):
                    st.image(cv2.cvtColor(result["ann_board"], cv2.COLOR_BGR2RGB),
                             use_container_width=True)
            else:
                st.info("⚠️ Board outline not detected — showing raw frame detections. "
                        "Try a photo with clearer board edges, or adjust the Confidence slider.")
        show_results(result)


# ── Video mode ────────────────────────────────────────────────────────────────
elif mode == "Video":
    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv"],
        key="vid",
    )
    if uploaded:
        with tempfile.NamedTemporaryFile(delete=False,
                                         suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        cap        = cv2.VideoCapture(tmp_path)
        total      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps        = cap.get(cv2.CAP_PROP_FPS) or 30
        # Run detection every STEP frames (≈ 5 times per second)
        step       = max(1, int(fps / 5))

        placeholder  = mid.empty()
        status_txt   = mid.empty()
        best_result  = None   # result with the most pieces detected so far
        last_result  = None   # result from the most recent processed frame
        frame_idx    = 0
        processed_n  = 0      # how many frames have been processed

        # Placeholder in right column — replaced every VIDEO_BOARD_UPDATE_EVERY frames
        right_panel = right.empty()

        progress = st.progress(0)

        while cap.isOpened():
            ok, frame = cap.read()
            if not ok:
                break
            frame_idx += 1

            # Update progress bar
            if total > 0:
                progress.progress(min(frame_idx / total, 1.0))

            # Skip frames between detection intervals
            if frame_idx % step != 0:
                continue

            status_txt.caption(f"Processing frame {frame_idx} / {total} …")
            last_result = process(frame)
            processed_n += 1

            # Keep the result with the most pieces as the "best" board state.
            # This ignores frames where a hand is covering the board.
            if (best_result is None or
                    len(last_result["detections"]) > len(best_result["detections"])):
                best_result = last_result

            # Always show the current annotated frame so the user sees live progress
            placeholder.image(
                cv2.cvtColor(last_result["ann_frame"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption=f"Frame {frame_idx}",
            )

            # Refresh right-panel board state every N processed frames
            if processed_n % config.VIDEO_BOARD_UPDATE_EVERY == 0:
                current_best = best_result or last_result
                if current_best:
                    _write_board_results(current_best, right_panel)

        cap.release()
        progress.empty()
        status_txt.empty()

        # Show the frame with the most pieces in the final display
        final = best_result or last_result
        if final:
            placeholder.image(
                cv2.cvtColor(final["ann_frame"], cv2.COLOR_BGR2RGB),
                use_container_width=True,
                caption="Best detection frame",
            )
            show_results(final)


# ── Webcam live mode ──────────────────────────────────────────────────────────
elif mode == "Webcam (live)":
    # Release camera when user switches away from live mode
    if not st.session_state.get("webcam_on") and "live_cap" in st.session_state:
        st.session_state.pop("live_cap").release()
        for _k in ("bg_sub", "last_quad", "live_dets", "frame_n", "live_board"):
            st.session_state.pop(_k, None)

    with left:
        detect_every = st.slider(
            "YOLO every N frames", 1, 8, 3, key="live_every",
            help="Lower = more accurate, higher = faster. Board outline updates every frame.",
        )

    with mid:
        st.subheader("Live feed")
        c1, c2 = st.columns(2)
        # Buttons live OUTSIDE the fragment so they trigger full-page reruns,
        # which allows the right-column board state to refresh on Start/Stop.
        if c1.button("▶ Start", key="live_start"):
            if "live_cap" not in st.session_state:
                st.session_state.live_cap  = cv2.VideoCapture(0)
                st.session_state.bg_sub    = cv2.createBackgroundSubtractorMOG2(
                    history=120, varThreshold=40, detectShadows=True
                )
                st.session_state.frame_n   = 0
                st.session_state.last_quad = None
                st.session_state.live_dets = []
            st.session_state.webcam_on = True

        if c2.button("⏹ Stop", key="live_stop"):
            st.session_state.webcam_on = False
            cap_obj = st.session_state.pop("live_cap", None)
            if cap_obj:
                cap_obj.release()
            for _k in ("bg_sub", "last_quad", "live_dets", "frame_n", "live_board"):
                st.session_state.pop(_k, None)

        # ── Fragment: only this region re-renders every 50 ms ─────────────────
        # Full-page flicker is eliminated because st.fragment rerenders only
        # its own content, leaving the sidebar, buttons, and right column stable.
        @st.fragment(run_every=0.05)
        def _live_fragment():
            if not st.session_state.get("webcam_on") or "live_cap" not in st.session_state:
                st.info(
                    "▶ Click **Start** to begin.  \n"
                    "The green outline will follow the board's tilt in real time. "
                    "Moving pieces are highlighted in amber."
                )
                return

            cap    = st.session_state.live_cap
            bg_sub = st.session_state.bg_sub

            frame_n = st.session_state.get("frame_n", 0) + 1
            st.session_state.frame_n = frame_n
            detect_every_n = st.session_state.get("live_every", 3)

            ret, frame = cap.read()
            if not ret:
                st.error("⚠️ Cannot read from webcam — check that no other app is using it.")
                return

            # ── Board detection (Canny/contour — fast, every frame) ──────────
            quad = cv_pipeline.detect_board(frame)
            if quad is not None:
                st.session_state.last_quad = quad
            last_quad = st.session_state.get("last_quad")

            # ── YOLO tracking (heavy — frame 1 and every N frames) ─────────
            # Run on frame 1 so there's no initial delay before pieces appear.
            if frame_n == 1 or frame_n % detect_every_n == 0:
                board_img = cv_pipeline.rectify_board(frame, last_quad)
                dets = cv_pipeline.detect_pieces(
                    frame,
                    board_img if last_quad is not None else None,
                    track=True,   # ByteTrack — persistent piece IDs across frames
                )
                st.session_state.live_dets = dets
            last_dets = st.session_state.get("live_dets", [])

            # ── Motion detection (MOG2 — every frame) ─────────────────────
            motion_mask = cv_pipeline.detect_motion(frame, bg_sub)

            # ── Draw results ───────────────────────────────────────────
            board_img_draw       = cv_pipeline.rectify_board(frame, last_quad)
            ann_frame, ann_board = cv_pipeline.draw_results(
                frame, last_quad, board_img_draw, last_dets
            )

            # Amber tint on moving regions (hands, pieces being picked up)
            if motion_mask is not None and motion_mask.any():
                tint = ann_frame.copy()
                tint[motion_mask > 0] = (0, 140, 255)
                ann_frame = cv2.addWeighted(ann_frame, 0.82, tint, 0.18, 0)

            status = (
                f"Frame {frame_n} · "
                + ("Board detected" if last_quad is not None else "Searching for board…")
                + f" · {len(last_dets)} piece{'s' if len(last_dets) != 1 else ''} detected"
            )

            # Two-column layout inside the fragment so image and board state
            # both update every 50 ms without needing a full-page rerun.
            img_col, state_col = st.columns([3, 2])

            with img_col:
                st.image(cv2.cvtColor(ann_frame, cv2.COLOR_BGR2RGB),
                         use_container_width=True, caption=status)
                if last_quad is not None and ann_board is not None:
                    with st.expander("Corrected board view", expanded=False):
                        st.image(cv2.cvtColor(ann_board, cv2.COLOR_BGR2RGB),
                                 use_container_width=True)

            with state_col:
                board_state = cv_pipeline.create_board_state(last_dets, frame, last_quad)
                board_status = "Board detected" if last_quad is not None else "Board not found"
                st.caption(f"{board_status} · {len(last_dets)} piece{'s' if len(last_dets) != 1 else ''} detected")
                st.subheader("Board state")
                st.code(board_to_text(board_state))
                st.subheader("FEN")
                st.code(board_to_fen(board_state))

        _live_fragment()

    with right:
        st.caption("Board state updates live in the centre panel.")


# ── Webcam snapshot mode ──────────────────────────────────────────────────────
elif mode == "Webcam snapshot":
    with mid:
        st.info("Point the camera at the board, then click the button below.")
        snapshot = st.camera_input("Capture frame")
        if snapshot:
            frame  = cv_pipeline.load_input(snapshot.read())
            result = process(frame)
            st.subheader("Detected pieces")
            st.image(cv2.cvtColor(result["ann_frame"], cv2.COLOR_BGR2RGB),
                     use_container_width=True)
            with st.expander("Corrected board view"):
                st.image(cv2.cvtColor(result["ann_board"], cv2.COLOR_BGR2RGB),
                         use_container_width=True)
            show_results(result)

else:
    with mid:
        st.info("Choose an input source on the left to begin.")
