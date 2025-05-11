#
# Author: Liwei Liao
# Contact: levio.pku@gmail.com
#

import os
import sys

sys.path.append("./gs/")
sys.path.append("./i3dv_interface/")
import time
import torch
from GPUtil import getGPUs
from simple_render import render_simple
from gs.scene.gaussian_model import GaussianModel
# from gs.scene import Scene
import uuid
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
from cam_utils import OrbitCamera, MiniCam, load_obj, save_obj
import numpy as np
import dearpygui.dearpygui as dpg
import datetime
from PIL import Image
from i3dv_interface.dynamic_scene import SceneI3DV

try:
    from torch.utils.tensorboard import SummaryWriter

    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False


def complete_ply_path(path):
    return path if path.endswith(".ply") else f"{path}/point_cloud/iteration_30000/point_cloud.ply"


class GUI:
    def __init__(self, args, dataset, opt, pipe) -> None:
        self.dataset = dataset
        self.args = args
        self.opt = opt
        self.pipe = pipe
        self.tb_writer = prepare_output_and_logger(dataset)

        self.need_update = False

        self.is_change_gau = False
        self.scene_path = self.args.model_path
        self.model = SceneI3DV(self.scene_path,
                               H=self.args.H, W=self.args.W)

        self.bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        # For UI
        self.visualization_mode = 'RGB'
        self.W = args.W
        self.H = args.H
        self.cam = OrbitCamera(args.W, args.H, r=args.radius, fovy=args.fovy)
        self.mode = "RGB"
        self.buffer_image = np.ones((self.W, self.H, 3), dtype=np.float32)

        # For Screenshot
        self.should_save_screenshot = False
        self.screenshot_id = 0
        self.screenshot_sv_path = f'./screenshot/' + datetime.datetime.now().strftime('%Y-%m-%d')

        # Dynamic
        self.video_len = len(self.model)
        self.fid = 0

        dpg.create_context()
        self.register_dpg()
        self.test_step()

    def __del__(self):
        dpg.destroy_context()

    def register_dpg(self):
        ### register texture
        with dpg.texture_registry(show=False):
            dpg.add_raw_texture(
                self.W, self.H,
                self.buffer_image,
                format=dpg.mvFormat_Float_rgb,
                tag="_texture",
            )

        ### register window
        with dpg.window(
                tag="_primary_window", width=self.W, height=self.H,
                pos=[0, 0], no_move=True, no_title_bar=True,
                no_scrollbar=True,
        ):
            dpg.add_image("_texture")

        # control window
        with dpg.window(
                label="Control", tag="_control_window",
                width=600, height=self.H, pos=[self.W, 0],
                no_move=True, no_title_bar=True,
        ):
            # button theme
            with dpg.theme() as theme_button:
                with dpg.theme_component(dpg.mvButton):
                    dpg.add_theme_color(dpg.mvThemeCol_Button, (23, 3, 18))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (51, 3, 47))
                    dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (83, 18, 83))
                    dpg.add_theme_style(dpg.mvStyleVar_FrameRounding, 5)
                    dpg.add_theme_style(dpg.mvStyleVar_FramePadding, 3, 3)

            # performance stuff
            with dpg.group(horizontal=True):
                dpg.add_text("no data", tag="_log_infer_time")
                dpg.add_text("no data", tag="_log_GPU_memory")

            # input stuff
            def callback_select_input(sender, app_data):
                self.need_update = True
                ply_path = complete_ply_path(app_data['file_path_name'])
                assert os.path.exists(ply_path)
                self.ply_path = ply_path
                print(f"New ply: {self.ply_path}")
                self.is_change_gau = True

            with dpg.file_dialog(
                    directory_selector=False,
                    show=False,
                    callback=callback_select_input,
                    file_count=1,
                    tag="change_path",
                    width=700,
                    height=400,
            ):
                dpg.add_file_extension("Ply{.ply}")

            with dpg.group(horizontal=True):
                dpg.add_button(
                    label="Change Path",
                    callback=lambda: dpg.show_item("change_path"),
                )
                dpg.add_text(self.scene_path, tag="Model_Path")

            with dpg.collapsing_header(label="User Guide", default_open=False):
                dpg.add_text("Press [Esc] to exit.", tag="Guide")

            # rendering options
            with dpg.collapsing_header(label="Rendering", default_open=True):
                # mode combo
                def callback_change_mode(sender, app_data):
                    self.mode = app_data
                    self.need_update = True

                dpg.add_combo(
                    ("RGB", "depth", "alpha", "normal_dep"),
                    label="mode",
                    default_value=self.mode,
                    callback=callback_change_mode,
                )

                # fov slider
                def callback_set_fovy(sender, app_data):
                    self.cam.fovy = np.deg2rad(app_data)
                    self.need_update = True

                dpg.add_slider_int(
                    label="FoV (vertical)",
                    min_value=1,
                    max_value=120,
                    format="%d deg",
                    default_value=np.rad2deg(self.cam.fovy),
                    callback=callback_set_fovy,
                )

                with dpg.group(horizontal=True):
                    dpg.add_text("Background:")

                    def callback_set_bg_red(sender, app_data):
                        # Clamp value between 0 and 1
                        self.bg_color[0] = max(0.0, min(1.0, app_data))
                        self.need_update = True
                        # Update the input field to show clamped value
                        dpg.set_value("_input_bg_red", self.bg_color[0])

                    def callback_set_bg_green(sender, app_data):
                        self.bg_color[1] = max(0.0, min(1.0, app_data))
                        self.need_update = True
                        dpg.set_value("_input_bg_green", self.bg_color[1])

                    def callback_set_bg_blue(sender, app_data):
                        self.bg_color[2] = max(0.0, min(1.0, app_data))
                        self.need_update = True
                        dpg.set_value("_input_bg_blue", self.bg_color[2])

                    # Red channel
                    dpg.add_input_float(
                        label="R", min_value=0.0, max_value=1.0,
                        default_value=self.bg_color[0],
                        callback=callback_set_bg_red,
                        width=80,
                        tag="_input_bg_red",
                        step=0.01,
                        format="%.2f"
                    )

                    # Green channel
                    dpg.add_input_float(
                        label="G", min_value=0.0, max_value=1.0,
                        default_value=self.bg_color[1],
                        callback=callback_set_bg_green,
                        width=80,
                        tag="_input_bg_green",
                        step=0.01,
                        format="%.2f"
                    )

                    # Blue channel
                    dpg.add_input_float(
                        label="B", min_value=0.0, max_value=1.0,
                        default_value=self.bg_color[2],
                        callback=callback_set_bg_blue,
                        width=80,
                        tag="_input_bg_blue",
                        step=0.01,
                        format="%.2f"
                    )

                # save screenshot
                with dpg.group(horizontal=True):
                    def callback_screenshot(sender, app_data):
                        self.should_save_screenshot = True

                    dpg.add_button(
                        label="screenshot", tag="_button_screenshot", callback=callback_screenshot
                    )
                    dpg.bind_item_theme("_button_screenshot", theme_button)

        def callback_set_mouse_loc(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return
            self.mouse_loc = np.array(app_data)

        def callback_camera_drag_rotate_or_draw_mask(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.orbit(dx, dy)
            self.need_update = True

        def callback_camera_wheel_scale(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            delta = app_data

            self.cam.scale(delta)
            self.need_update = True

        def callback_camera_drag_pan(sender, app_data):
            if not dpg.is_item_focused("_primary_window"):
                return

            dx = app_data[1]
            dy = app_data[2]

            self.cam.pan(dx, dy)
            self.need_update = True

        with dpg.handler_registry():
            # for camera moving
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Left,
                callback=callback_camera_drag_rotate_or_draw_mask,
            )
            dpg.add_mouse_wheel_handler(callback=callback_camera_wheel_scale)
            dpg.add_mouse_drag_handler(
                button=dpg.mvMouseButton_Middle, callback=callback_camera_drag_pan
            )

            dpg.add_mouse_move_handler(callback=callback_set_mouse_loc)

        dpg.create_viewport(
            title="DearGaussian",
            width=self.W + 600,
            height=self.H + (45 if os.name == "nt" else 0),
            resizable=False,
            small_icon="assets/pink_icon.png"
        )

        ### global theme
        with dpg.theme() as theme_no_padding:
            with dpg.theme_component(dpg.mvAll):
                # set all padding to 0 to avoid scroll bar
                dpg.add_theme_style(
                    dpg.mvStyleVar_WindowPadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_FramePadding, 0, 0, category=dpg.mvThemeCat_Core
                )
                dpg.add_theme_style(
                    dpg.mvStyleVar_CellPadding, 0, 0, category=dpg.mvThemeCat_Core
                )

        dpg.bind_item_theme("_primary_window", theme_no_padding)

        dpg.setup_dearpygui()

        if os.path.exists("LXGWWenKai-Regular.ttf"):
            with dpg.font_registry():
                with dpg.font("LXGWWenKai-Regular.ttf", 18) as default_font:
                    dpg.bind_font(default_font)

        dpg.show_viewport()

    # gui mode
    def render(self):
        while dpg.is_dearpygui_running():
            self.test_step()
            dpg.render_dearpygui_frame()

    @torch.no_grad()
    def test_step(self, specified_cam=None):
        if dpg.is_key_down(dpg.mvKey_Escape):
            print("Exit GUI!")
            dpg.stop_dearpygui()

        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
        starter.record()

        if self.should_save_screenshot and os.path.exists(
                os.path.join(self.args.model_path, 'screenshot_camera.pickle')):
            print('Use fixed camera for screenshot: ', os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
            cur_cam = load_obj(os.path.join(self.args.model_path, 'screenshot_camera.pickle'))
        elif specified_cam is not None:
            cur_cam = specified_cam
        else:
            cur_cam = MiniCam(
                self.cam.pose,
                self.W,
                self.H,
                self.cam.fovy,
                self.cam.fovx,
                self.cam.near,
                self.cam.far,
                fid=0
            )

        # rendering step
        out = self.model.spatial_temporal_render(cur_cam, self.fid)

        # loop play
        self.fid = self.fid + 1 if self.fid < self.video_len - 1 else 0

        if self.mode == "normal_dep":
            from cam_utils import depth2normal
            normal = depth2normal(out["depth"])
            out["normal_dep"] = (normal + 1) / 2

        buffer_image = out["render"
        if self.mode == "RGB" else self.mode]  # [3, H, W]

        if self.should_save_screenshot:
            alpha = out['alpha']
            sv_image = torch.cat([buffer_image, alpha], dim=0).clamp(0, 1).permute(1, 2, 0).detach().cpu().numpy()

            def save_image(image, image_dir):
                os.makedirs(image_dir, exist_ok=True)
                idx = len(os.listdir(image_dir))
                print('>>> Saving image to %s' % os.path.join(image_dir, '%05d.png' % idx))
                Image.fromarray((image * 255).astype('uint8')).save(os.path.join(image_dir, '%05d.png' % idx))
                # Save the camera of screenshot
                save_obj(os.path.join(image_dir, '%05d_cam.pickle' % idx), cur_cam)

            save_image(sv_image, self.screenshot_sv_path)
            self.should_save_screenshot = False

        if self.mode in ['depth', 'alpha']:
            buffer_image = buffer_image.repeat(3, 1, 1)
            if self.mode == 'depth':
                buffer_image = (buffer_image - buffer_image.min()) / (buffer_image.max() - buffer_image.min() + 1e-20)

        buffer_image = torch.nn.functional.interpolate(
            buffer_image.unsqueeze(0),
            size=(self.H, self.W),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

        self.buffer_image = (
            buffer_image.permute(1, 2, 0)
            .contiguous()
            .clamp(0, 1)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
        )

        ender.record()
        torch.cuda.synchronize()
        t = starter.elapsed_time(ender)

        dpg.set_value("_log_infer_time", f"FPS: {int(1000 / t)} (Infer time: {t:.2f}ms) ")
        mem_value = getGPUs()[0].memoryUsed
        dpg.set_value("_log_GPU_memory", f"GPU Memory: {mem_value} MB")
        dpg.set_value("_texture", self.buffer_image)
        dpg.set_value("Model_Path", self.scene_path)
        self.background = torch.tensor(self.bg_color, dtype=torch.float32, device="cuda")

        if self.is_change_gau:
            # self.gaussians = GaussianModel(3)
            # self.gaussians.load_ply(self.ply_path)
            self.is_change_gau = False

        return self.buffer_image


def prepare_output_and_logger(args):
    print("Scene Folder: {}".format(args.model_path))
    os.makedirs("output", exist_ok=True)
    with open(os.path.join("output", "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter("output")
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="GUI Parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--W', type=int, default=800, help="GUI width")
    parser.add_argument('--H', type=int, default=800, help="GUI height")
    parser.add_argument('--radius', type=float, default=5, help="default GUI camera radius from center")
    parser.add_argument('--fovy', type=float, default=50, help="default GUI camera fovy")
    args = parser.parse_args(sys.argv[1:])

    gui = GUI(args=args, dataset=lp.extract(args), opt=op.extract(args), pipe=pp.extract(args))
    gui.render()
