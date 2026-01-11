from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import gradio as gr
import numpy as np

from .utils import (
    GRADIO_VERSION,
    gen_examples,
    get_matcher_zoo,
    load_config,
    ransac_zoo,
    run_matching,
    run_ransac,
    run_generate_warp_images,
    send_to_match,
)
from matching import available_models

DESCRIPTION = """
<div style="text-align: left;">
  <h1 style="font-size: 2rem; font-weight: bold; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; margin-bottom: 1rem;">
    Image Matching WebUI
  </h1>
</div>

This Space demonstrates [Image Matching WebUI](https://github.com/Vincentqyw/image-matching-webui) by vincent qin. Feel free to play with it, or duplicate to run image matching without a queue!
<br/>
🚀 **Now GPU-accelerated!** Thanks to HuggingFace's community grant, all algorithms run on GPU for fast, responsive inference.

🔎 For more details about supported local features and matchers, please refer to https://github.com/Vincentqyw/image-matching-webui

🐛 Your feedback is valuable to me. Please do not hesitate to report any bugs [here](https://github.com/Vincentqyw/image-matching-webui/issues).
"""

CSS = """
#warning {background-color: #FFCCCB}
.logs_class textarea {font-size: 10px !important}

.gradio-container {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}

"""


class ImageMatchingApp:
    def __init__(self, server_name="0.0.0.0", server_port=7860, **kwargs):
        self.server_name = server_name
        self.server_port = server_port
        self.config_path = kwargs.get("config", Path(__file__).parent / "config.yaml")
        self.cfg = load_config(self.config_path)
        self.matcher_zoo = get_matcher_zoo(self.cfg["matcher_zoo"])
        self.app = None
        self.example_data_root = kwargs.get(
            "example_data_root", Path(__file__).parents[1] / "datasets"
        )
        # final step
        self.init_interface()

    def init_matcher_dropdown(self):
        return available_models

    def init_interface(self):
        with gr.Blocks(css=CSS) as self.app:
            with gr.Tab("Image Matching"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Image(
                            str(Path(__file__).parent.parent / "assets/logo.webp"),
                            elem_id="logo-img",
                            show_label=False,
                            show_share_button=False,
                            show_download_button=False,
                        )
                    with gr.Column(scale=3):
                        gr.Markdown(DESCRIPTION)
                with gr.Row(equal_height=False):
                    with gr.Column():
                        with gr.Row():
                            matcher_list = gr.Dropdown(
                                choices=self.init_matcher_dropdown(),
                                value="disk-lg",
                                label="Matching Model",
                                interactive=True,
                            )
                            match_image_src = gr.Radio(
                                (
                                    ["upload", "webcam", "clipboard"]
                                    if GRADIO_VERSION > "3"
                                    else ["upload", "webcam", "canvas"]
                                ),
                                label="Image Source",
                                value="upload",
                            )
                        with gr.Row():
                            input_image0 = gr.Image(
                                label="Image 0",
                                type="numpy",
                                image_mode="RGB",
                                height=300 if GRADIO_VERSION > "3" else None,
                                interactive=True,
                            )
                            input_image1 = gr.Image(
                                label="Image 1",
                                type="numpy",
                                image_mode="RGB",
                                height=300 if GRADIO_VERSION > "3" else None,
                                interactive=True,
                            )

                        with gr.Row():
                            button_reset = gr.Button(value="Reset")
                            button_run = gr.Button(value="Run Match", variant="primary")
                        with gr.Row():
                            button_stop = gr.Button(value="Force Stop", variant="stop")

                        with gr.Accordion("Advanced Setting", open=False):
                            with gr.Tab("Matching Setting"):
                                with gr.Row():
                                    match_setting_threshold = gr.Slider(
                                        minimum=0.0,
                                        maximum=1,
                                        step=0.001,
                                        label="Match threshold",
                                        value=0.1,
                                    )
                                    match_setting_max_keypoints = gr.Slider(
                                        minimum=10,
                                        maximum=10000,
                                        step=10,
                                        label="Max features",
                                        value=1000,
                                    )
                                # TODO: add line settings
                                with gr.Row():
                                    detect_keypoints_threshold = gr.Slider(
                                        minimum=0,
                                        maximum=1,
                                        step=0.001,
                                        label="Keypoint threshold",
                                        value=0.015,
                                    )
                                    detect_line_threshold = gr.Slider(  # noqa: F841
                                        minimum=0.1,
                                        maximum=1,
                                        step=0.01,
                                        label="Line threshold",
                                        value=0.2,
                                    )

                            with gr.Tab("RANSAC Setting"):
                                with gr.Row(equal_height=False):
                                    ransac_method = gr.Dropdown(
                                        choices=ransac_zoo.keys(),
                                        value=self.cfg["defaults"]["ransac_method"],
                                        label="RANSAC Method",
                                        interactive=True,
                                    )
                                ransac_reproj_threshold = gr.Slider(
                                    minimum=0.0,
                                    maximum=12,
                                    step=0.01,
                                    label="Ransac Reproj threshold",
                                    value=8.0,
                                )
                                ransac_confidence = gr.Slider(
                                    minimum=0.0,
                                    maximum=1,
                                    step=0.00001,
                                    label="Ransac Confidence",
                                    value=self.cfg["defaults"]["ransac_confidence"],
                                )
                                ransac_max_iter = gr.Slider(
                                    minimum=0.0,
                                    maximum=100000,
                                    step=100,
                                    label="Ransac Iterations",
                                    value=self.cfg["defaults"]["ransac_max_iter"],
                                )
                                button_ransac = gr.Button(
                                    value="Rerun RANSAC", variant="primary"
                                )
                            with gr.Tab("Geometry Setting"):
                                with gr.Row(equal_height=False):
                                    choice_geometry_type = gr.Radio(
                                        ["Fundamental", "Homography"],
                                        label="Reconstruct Geometry",
                                        value=self.cfg["defaults"]["setting_geometry"],
                                    )
                            with gr.Tab("Image Setting"):
                                with gr.Row():
                                    image_force_resize_cb = gr.Checkbox(
                                        label="Force Resize",
                                        value=False,
                                        interactive=True,
                                    )
                                    image_setting_height = gr.Slider(
                                        minimum=48,
                                        maximum=2048,
                                        step=16,
                                        label="Image Height",
                                        value=480,
                                        visible=False,
                                    )
                                    image_setting_width = gr.Slider(
                                        minimum=64,
                                        maximum=2048,
                                        step=16,
                                        label="Image Width",
                                        value=640,
                                        visible=False,
                                    )
                        # image resize
                        image_force_resize_cb.select(
                            fn=self._on_select_force_resize,
                            inputs=image_force_resize_cb,
                            outputs=[image_setting_width, image_setting_height],
                        )
                        # collect inputs
                        state_cache = gr.State({})
                        inputs = [
                            input_image0,
                            input_image1,
                            match_setting_threshold,
                            match_setting_max_keypoints,
                            detect_keypoints_threshold,
                            matcher_list,
                            ransac_method,
                            ransac_reproj_threshold,
                            ransac_confidence,
                            ransac_max_iter,
                            choice_geometry_type,
                            gr.State(self.matcher_zoo),
                            image_force_resize_cb,
                            image_setting_width,
                            image_setting_height,
                        ]

                        # Add some examples
                        with gr.Row():
                            # Example inputs
                            with gr.Accordion("Open for More: Examples", open=True):
                                gr.Examples(
                                    examples=gen_examples(self.example_data_root),
                                    inputs=inputs,
                                    outputs=[],
                                    fn=run_matching,
                                    cache_examples=False,
                                    label=(
                                        "Examples (click one of the images below to Run"
                                        " Match). Thx: WxBS"
                                    ),
                                )
                        # TODO: add a table of supported algorithms
                        # with gr.Accordion("Supported Algorithms", open=False):
                        #     # add a table of supported algorithms
                        #     self.display_supported_algorithms()

                    with gr.Column():
                        with gr.Accordion("Open for More: Keypoints", open=True):
                            output_keypoints = gr.Image(label="Keypoints", type="numpy")
                        with gr.Accordion(
                            (
                                "Open for More: Raw Matches"
                                " (Green for good matches, Red for bad)"
                            ),
                            open=False,
                        ):
                            output_matches_raw = gr.Image(
                                label="Raw Matches",
                                type="numpy",
                            )
                        with gr.Accordion(
                            (
                                "Open for More: Ransac Matches"
                                " (Green for good matches, Red for bad)"
                            ),
                            open=True,
                        ):
                            output_matches_ransac = gr.Image(
                                label="Ransac Matches", type="numpy"
                            )
                        with gr.Accordion(
                            "Open for More: Matches Statistics", open=False
                        ):
                            output_pred = gr.File(label="Outputs", elem_id="download")
                            matches_result_info = gr.JSON(label="Matches Statistics")
                            matcher_info = gr.JSON(label="Match info")

                        with gr.Accordion("Open for More: Warped Image", open=True):
                            output_wrapped = gr.Image(
                                label="Wrapped Pair", type="numpy"
                            )
                            # send to input
                            button_rerun = gr.Button(
                                value="Send to Input Match Pair",
                                variant="primary",
                            )
                            with gr.Accordion(
                                "Open for More: Geometry info", open=False
                            ):
                                geometry_result = gr.JSON(
                                    label="Reconstructed Geometry"
                                )

                    # callbacks
                    match_image_src.change(
                        fn=self.ui_change_imagebox,
                        inputs=match_image_src,
                        outputs=input_image0,
                    )
                    match_image_src.change(
                        fn=self.ui_change_imagebox,
                        inputs=match_image_src,
                        outputs=input_image1,
                    )
                    # collect outputs
                    outputs = [
                        output_keypoints,
                        output_matches_raw,
                        output_matches_ransac,
                        matches_result_info,
                        matcher_info,
                        geometry_result,
                        output_wrapped,
                        state_cache,
                        output_pred,
                    ]
                    # button callbacks
                    click_event = button_run.click(
                        fn=run_matching, inputs=inputs, outputs=outputs
                    )
                    # stop button
                    button_stop.click(
                        fn=None, inputs=None, outputs=None, cancels=[click_event]
                    )

                    # Reset images
                    reset_outputs = [
                        input_image0,
                        input_image1,
                        match_setting_threshold,
                        match_setting_max_keypoints,
                        detect_keypoints_threshold,
                        matcher_list,
                        input_image0,
                        input_image1,
                        match_image_src,
                        output_keypoints,
                        output_matches_raw,
                        output_matches_ransac,
                        matches_result_info,
                        matcher_info,
                        output_wrapped,
                        geometry_result,
                        ransac_method,
                        ransac_reproj_threshold,
                        ransac_confidence,
                        ransac_max_iter,
                        choice_geometry_type,
                        output_pred,
                        image_force_resize_cb,
                    ]
                    button_reset.click(
                        fn=self.ui_reset_state,
                        inputs=None,
                        outputs=reset_outputs,
                    )

                    # run ransac button action
                    button_ransac.click(
                        fn=run_ransac,
                        inputs=[
                            state_cache,
                            choice_geometry_type,
                            ransac_method,
                            ransac_reproj_threshold,
                            ransac_confidence,
                            ransac_max_iter,
                        ],
                        outputs=[
                            output_matches_ransac,
                            matches_result_info,
                            output_wrapped,
                            output_pred,
                        ],
                    )

                    # send warped image to match
                    button_rerun.click(
                        fn=send_to_match,
                        inputs=[state_cache],
                        outputs=[input_image0, input_image1],
                    )

                    # estimate geo
                    choice_geometry_type.change(
                        fn=run_generate_warp_images,
                        inputs=[
                            input_image0,
                            input_image1,
                            geometry_result,
                            choice_geometry_type,
                        ],
                        outputs=[output_wrapped, geometry_result],
                    )

    def run(self):
        self.app.queue().launch(
            server_name=self.server_name,
            server_port=self.server_port,
            share=False,
            allowed_paths=[
                str(Path(__file__).parents[0]),
                str(Path(__file__).parents[1]),
                str(self.example_data_root),
            ],
        )

    def ui_change_imagebox(self, choice):
        """
        Updates the image box with the given choice.

        Args:
            choice (list): The list of image sources to be displayed in the image box.

        Returns:
            dict: A dictionary containing the updated value, sources, and type for the image box.
        """
        ret_dict = {
            "value": None,  # The updated value of the image box
            "__type__": "update",  # The type of update for the image box
        }
        if GRADIO_VERSION > "3":
            return {
                **ret_dict,
                "sources": choice,  # The list of image sources to be displayed
            }
        else:
            return {
                **ret_dict,
                "source": choice,  # The list of image sources to be displayed
            }

    def _on_select_force_resize(self, visible: bool = False):
        return gr.update(visible=visible), gr.update(visible=visible)

    def ui_reset_state(
        self,
        *args: Any,
    ) -> Tuple[
        Optional[np.ndarray],
        Optional[np.ndarray],
        float,
        int,
        float,
        str,
        Dict[str, Any],
        Dict[str, Any],
        str,
        Optional[np.ndarray],
        Optional[np.ndarray],
        Optional[np.ndarray],
        Dict[str, Any],
        Dict[str, Any],
        Optional[np.ndarray],
        Dict[str, Any],
        str,
        int,
        float,
        int,
        bool,
    ]:
        """
        Reset the state of the UI.

        Returns:
            tuple: A tuple containing the initial values for the UI state.
        """
        model_name: str = available_models[0]  # Get the first key from matcher_zoo
        # flush_logs()
        return (
            None,  # image0: Optional[np.ndarray]
            None,  # image1: Optional[np.ndarray]
            self.cfg["defaults"]["match_threshold"],  # matching_threshold: float
            self.cfg["defaults"]["max_keypoints"],  # max_keypoints: int
            self.cfg["defaults"]["keypoint_threshold"],  # keypoint_threshold: float
            model_name,  # matcher: str
            self.ui_change_imagebox("upload"),  # input image0: Dict[str, Any]
            self.ui_change_imagebox("upload"),  # input image1: Dict[str, Any]
            "upload",  # match_image_src: str
            None,  # keypoints: Optional[np.ndarray]
            None,  # raw matches: Optional[np.ndarray]
            None,  # ransac matches: Optional[np.ndarray]
            {},  # matches result info: Dict[str, Any]
            {},  # matcher config: Dict[str, Any]
            None,  # warped image: Optional[np.ndarray]
            {},  # geometry result: Dict[str, Any]
            self.cfg["defaults"]["ransac_method"],  # ransac_method: str
            self.cfg["defaults"][
                "ransac_reproj_threshold"
            ],  # ransac_reproj_threshold: float
            self.cfg["defaults"]["ransac_confidence"],  # ransac_confidence: float
            self.cfg["defaults"]["ransac_max_iter"],  # ransac_max_iter: int
            self.cfg["defaults"]["setting_geometry"],  # geometry: str
            None,  # predictions
            False,
        )

    def display_supported_algorithms(self, style="tab"):
        def get_link(link, tag="Link"):
            return "[{}]({})".format(tag, link) if link is not None else "None"

        data = []
        cfg = self.cfg["matcher_zoo"]
        if style == "md":
            markdown_table = "| Algo. | Conference | Code | Project | Paper |\n"
            markdown_table += "| ----- | ---------- | ---- | ------- | ----- |\n"

            for _, v in cfg.items():
                if not v["info"].get("display", True):
                    continue
                github_link = get_link(v["info"].get("github", ""))
                project_link = get_link(v["info"].get("project", ""))
                paper_link = get_link(
                    v["info"]["paper"],
                    (
                        Path(v["info"]["paper"]).name[-10:]
                        if v["info"]["paper"] is not None
                        else "Link"
                    ),
                )

                markdown_table += "{}|{}|{}|{}|{}\n".format(
                    v["info"].get("name", ""),
                    v["info"].get("source", ""),
                    github_link,
                    project_link,
                    paper_link,
                )
            return gr.Markdown(markdown_table)
        elif style == "tab":
            for k, v in cfg.items():
                if not v["info"].get("display", True):
                    continue
                data.append(
                    [
                        v["info"].get("name", ""),
                        v["info"].get("source", ""),
                        v["info"].get("github", ""),
                        v["info"].get("paper", ""),
                        v["info"].get("project", ""),
                    ]
                )
            tab = gr.Dataframe(
                headers=["Algo.", "Conference", "Code", "Paper", "Project"],
                datatype=["str", "str", "str", "str", "str"],
                col_count=(5, "fixed"),
                value=data,
                # wrap=True,
                # min_width = 1000,
                # height=1000,
            )
            return tab
