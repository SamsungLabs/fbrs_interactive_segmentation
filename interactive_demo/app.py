import tkinter as tk
from tkinter import messagebox, filedialog, ttk

import cv2
import mxnet as mx
import numpy as np
from mxnet import gluon
from PIL import Image

from interactive_demo.wrappers import BoundedNumericalEntry, FocusHorizontalScale, FocusCheckButton, \
    FocusButton, FocusLabelFrame
from isegm.inference.predictors import get_predictor
from isegm.utils import vis
from isegm.inference import clicker
from interactive_demo.canvas import CanvasImage


class InteractiveDemoApp(ttk.Frame):
    def __init__(self, master, args, model):
        super().__init__(master)
        self.master = master
        master.title("Interactive Segmentation with f-BRS")
        master.withdraw()
        master.update_idletasks()
        x = (master.winfo_screenwidth() - master.winfo_reqwidth()) / 2
        y = (master.winfo_screenheight() - master.winfo_reqheight()) / 2
        master.geometry("+%d+%d" % (x, y))
        self.pack(fill="both", expand=True)

        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']

        self._init_state(num_max_points=args.n_clicks)
        self._add_menu()
        self._add_canvas()
        self._add_buttons()

        self.net = model
        self.ctx = args.ctx
        self.predictor = None
        self.zoomin_params = None
        self._change_zoomin()
        master.bind('<space>', lambda event: self._finish_object())

        self.state['zoomin_params']['skip_clicks'].trace(mode='w', callback=self._change_zoomin)
        self.state['zoomin_params']['target_size'].trace(mode='w', callback=self._change_zoomin)
        self.state['zoomin_params']['expansion_ratio'].trace(mode='w', callback=self._change_zoomin)
        self.state['predictor_params']['net_clicks_limit'].trace(mode='w', callback=self._change_brs_mode)
        self.state['lbfgs_max_iters'].trace(mode='w', callback=self._change_brs_mode)

        self.input_transform = gluon.data.vision.transforms.Compose([
            gluon.data.vision.transforms.ToTensor(),
            gluon.data.vision.transforms.Normalize([.485, .456, .406], [.229, .224, .225])
        ])

    def _init_state(self, num_max_points=20):
        self.state = {
            'zoomin_params': {
                'use_zoom_in': tk.BooleanVar(value=True),
                'skip_clicks': tk.IntVar(value=1),
                'target_size': tk.IntVar(value=480),
                'expansion_ratio': tk.DoubleVar(value=1.4)
            },
            '_zoomin_history': [],

            'predictor_params': {
                'num_max_points': num_max_points,
                'net_clicks_limit': tk.IntVar(value=8),
            },
            'brs_mode': tk.StringVar(value='f-BRS-B'),
            'prob_thresh': tk.DoubleVar(value=0.5),
            'lbfgs_max_iters': tk.IntVar(value=20),

            'alpha_blend': tk.DoubleVar(value=0.5),
            'click_radius': tk.IntVar(value=3),

            '_image': None,
            '_image_nd': None,

            '_pred_probs_history': [],
            '_object_counter': 0,
            '_object_masks': [],

            '_clicks_list': [],
            '_clicker': None,
        }

    def _reset_to_defaults(self):
        self.state.update({
            '_zoomin_history': [],
            '_clicks_list': [],
            '_pred_probs_history': [],
        })

        self.state['alpha_blend'].set(0.5)
        self.state['prob_thresh'].set(0.5)

        if self.state['_clicker'] is not None:
            self.state['_clicker'].reset_clicks()

        if self.predictor.zoom_in is not None:
            self.predictor.zoom_in.reset()

    def _add_menu(self):
        self.menubar = tk.Menu(self, bd=1, tearoff=False)
        self.menubar.add_command(label="Load image", command=self._load_image_callback)
        self.menubar.add_command(label="Save mask", command=self._save_mask_callback)

        self.menubar.add_command(label="About", command=self._about_callback)
        self.menubar.add_command(label="Exit", command=self.master.quit)

        self.master.config(menu=self.menubar)

    def _add_canvas(self):
        self.canvas_frame = FocusLabelFrame(self, text="Image")
        self.canvas_frame.rowconfigure(0, weight=1)
        self.canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(self.canvas_frame, highlightthickness=0, cursor="hand1", width=400, height=400)
        self.canvas.grid(row=0, column=0, sticky='nswe', padx=5, pady=5)

        self.image_on_canvas = None
        self.canvas_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=5, pady=5)

    def _add_buttons(self):
        self.control_frame = FocusLabelFrame(self, text="Controls")
        self.control_frame.pack(side=tk.TOP, fill='x', padx=5, pady=5)
        master = self.control_frame

        self.clicks_options_frame = FocusLabelFrame(master, text="Clicks management")
        self.clicks_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        self.finish_object_button = \
            FocusButton(self.clicks_options_frame, text='Finish\nobject', bg='#b6d7a8', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._finish_object)
        self.finish_object_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.undo_click_button = \
            FocusButton(self.clicks_options_frame, text='Undo click', bg='#ffe599', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._undo_click)
        self.undo_click_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)
        self.reset_clicks_button = \
            FocusButton(self.clicks_options_frame, text='Reset clicks', bg='#ea9999', fg='black', width=10, height=2,
                        state=tk.DISABLED, command=self._reset_clicks_history)
        self.reset_clicks_button.pack(side=tk.LEFT, fill=tk.X, padx=10, pady=3)

        self.zoomin_options_frame = FocusLabelFrame(master, text="ZoomIn options")
        self.zoomin_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusCheckButton(self.zoomin_options_frame, text='Use ZoomIn', command=self._change_zoomin,
                         variable=self.state['zoomin_params']['use_zoom_in']).grid(rowspan=3, column=0, padx=10)
        tk.Label(self.zoomin_options_frame, text="Skip clicks").grid(row=0, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Target size").grid(row=1, column=1, pady=1, sticky='e')
        tk.Label(self.zoomin_options_frame, text="Expand ratio").grid(row=2, column=1, pady=1, sticky='e')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['skip_clicks'],
                              min_value=0, max_value=None, vartype=int,
                              name='zoom_in_skip_clicks').grid(row=0, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['target_size'],
                              min_value=100, max_value=3000, vartype=int,
                              name='zoom_in_target_size').grid(row=1, column=2, padx=10, pady=1, sticky='w')
        BoundedNumericalEntry(self.zoomin_options_frame, variable=self.state['zoomin_params']['expansion_ratio'],
                              min_value=1.0, max_value=2.0, vartype=float,
                              name='zoom_in_expansion_ratio').grid(row=2, column=2, padx=10, pady=1, sticky='w')
        self.zoomin_options_frame.columnconfigure((0, 1, 2), weight=1)

        self.brs_options_frame = FocusLabelFrame(master, text="BRS options")
        self.brs_options_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        menu = tk.OptionMenu(self.brs_options_frame, self.state['brs_mode'],
                             *self.brs_modes, command=self._change_brs_mode)
        menu.config(width=11)
        menu.grid(rowspan=2, column=0, padx=10)
        self.net_clicks_label = tk.Label(self.brs_options_frame, text="Network clicks")
        self.net_clicks_label.grid(row=0, column=1, pady=2, sticky='e')
        self.net_clicks_entry = BoundedNumericalEntry(self.brs_options_frame,
                                                      variable=self.state['predictor_params']['net_clicks_limit'],
                                                      min_value=0, max_value=None, vartype=int, allow_inf=True,
                                                      name='net_clicks_limit')
        self.net_clicks_entry.grid(row=0, column=2, padx=10, pady=2, sticky='w')
        tk.Label(self.brs_options_frame, text="L-BFGS\nmax iterations").grid(row=1, column=1, pady=2, sticky='e')
        BoundedNumericalEntry(self.brs_options_frame, variable=self.state['lbfgs_max_iters'],
                              min_value=1, max_value=1000, vartype=int,
                              name='lbfgs_max_iters').grid(row=1, column=2, padx=10, pady=2, sticky='w')
        self.brs_options_frame.columnconfigure((0, 1), weight=1)

        self.prob_thresh_frame = FocusLabelFrame(master, text="Predictions threshold")
        self.prob_thresh_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.prob_thresh_frame, from_=0.0, to=1.0, command=self._update_prob_thresh,
                             variable=self.state['prob_thresh']).pack(padx=10)

        self.alpha_blend_frame = FocusLabelFrame(master, text="Alpha blending coefficient")
        self.alpha_blend_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.alpha_blend_frame, from_=0.0, to=1.0, command=self._update_blend_alpha,
                             variable=self.state['alpha_blend']).pack(padx=10, anchor=tk.CENTER)

        self.click_radius_frame = FocusLabelFrame(master, text="Visualisation click radius")
        self.click_radius_frame.pack(side=tk.TOP, fill=tk.X, padx=10, pady=3)
        FocusHorizontalScale(self.click_radius_frame, from_=0, to=7, resolution=1, command=self._update_click_radius,
                             variable=self.state['click_radius']).pack(padx=10, anchor=tk.CENTER)

    # ================================================= Menu callbacks =================================================
    def _load_image_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            filename = filedialog.askopenfilename(parent=self.master, filetypes=[
                ("Images", "*.jpg *.jpeg *.png *.bmp *.tiff"),
                ("All files", "*.*"),
            ], title="Chose an image")

            if len(filename) > 0:
                self.state['_image'] = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)
                self.state['_image_nd'] = self.input_transform(mx.ndarray.array(self.state['_image'], ctx=self.ctx))
                self.predictor.set_input_image(self.state['_image_nd'])
                self.state['_object_counter'] = 0
                self.state['_clicks_list'] = []
                self.state['_clicker'] = clicker.Clicker(np.zeros(self.state['_image'].shape[:2], dtype=np.bool))
                self.state['_pred_probs_history'] = []
                self.state['_zoomin_history'] = []
                self.state['_object_masks'] = []

                self._set_click_dependent_widgets_state()
                self._update_image(reset_canvas=True)

    def _save_mask_callback(self):
        self.menubar.focus_set()
        if self._check_entry(self):
            mask = self._get_current_mask()
            if mask is None:
                # messagebox.showwarning("Can't save an empty mask")
                return
            filename = filedialog.asksaveasfilename(parent=self.master, initialfile='mask.png', filetypes=[
                ("PNG image", "*.png"),
                ("BMP image", "*.bmp"),
                ("All files", "*.*"),
            ], title="Save current mask as...")

            if len(filename) > 0:
                cv2.imwrite(filename, mask)

    def _about_callback(self):
        self.menubar.focus_set()

        text = [
            "Developed by:",
            "K.Sofiiuk and I. Petrov",
            "MPL-2.0 License, 2020"
        ]

        messagebox.showinfo("About Demo", '\n'.join(text))

    # ================================================ Button callbacks ================================================
    def _finish_object(self):
        if len(self.state['_pred_probs_history']) == 0:
            return
        current_mask = (self.state['_pred_probs_history'][-1] > self.state['prob_thresh'].get()).astype(np.uint8)
        current_mask = current_mask * (self.state['_object_counter'] + 1)
        self.state['_object_masks'].append(current_mask)

        self.state['_object_counter'] += 1
        self.state['_clicks_list'] = []
        self.state['_clicker'].reset_clicks()
        self.state['_pred_probs_history'] = []
        self.state['_zoomin_history'] = []

        self._set_click_dependent_widgets_state()
        self._update_image()
        self._reset_predictor()

    def _undo_click(self):
        self.state['_clicks_list'] = self.state['_clicks_list'][:-1]
        self.state['_clicker']._remove_last_click()
        self.state['_pred_probs_history'] = self.state['_pred_probs_history'][:-1]
        self.state['_zoomin_history'] = self.state['_zoomin_history'][:-1]
        if len(self.state['_zoomin_history']) > 0:
            self.predictor.zoom_in._input_image = self.state['_zoomin_history'][-1]['_input_image']
            self.predictor.zoom_in._prev_probs = self.state['_zoomin_history'][-1]['_prev_probs']
            self.predictor.zoom_in._object_roi = self.state['_zoomin_history'][-1]['_object_roi']
            self.predictor.zoom_in._roi_image = self.state['_zoomin_history'][-1]['_roi_image']

        if len(self.state['_clicks_list']) == 0:
            self._reset_predictor()
            self._set_click_dependent_widgets_state()
        self._update_image()

    def _reset_clicks_history(self):
        self._reset_to_defaults()
        self._reset_predictor()

        self._set_click_dependent_widgets_state()
        self._update_image()

    # =============================================== Variable callbacks ===============================================
    def _update_prob_thresh(self, value):
        if len(self.state['_pred_probs_history']) > 0:
            self._update_image()

    def _update_blend_alpha(self, value):
        if self.state['_image'] is not None and \
                (len(self.state['_pred_probs_history']) > 0 or self.state['_object_counter'] > 0):
            self._update_image()

    def _update_click_radius(self, *args):
        if self.image_on_canvas is None:
            return

        self._update_image()

    def _change_zoomin(self, *args):
        if self.state['zoomin_params']['use_zoom_in'].get():
            self.zoomin_params = {
                'skip_clicks': self.state['zoomin_params']['skip_clicks'].get(),
                'target_size': self.state['zoomin_params']['target_size'].get(),
                'expansion_ratio': self.state['zoomin_params']['expansion_ratio'].get()
            }

            self._reset_predictor()

            if len(self.state['_zoomin_history']) > 0:
                self.predictor.zoom_in._input_image = self.state['_zoomin_history'][-1]['_input_image']
                self.predictor.zoom_in._prev_probs = self.state['_zoomin_history'][-1]['_prev_probs']
                self.predictor.zoom_in._object_roi = self.state['_zoomin_history'][-1]['_object_roi']
                self.predictor.zoom_in._roi_image = self.state['_zoomin_history'][-1]['_roi_image']
            elif self.predictor is not None:
                self.predictor.zoom_in.reset()
        else:
            self.zoomin_params = None
            self._reset_predictor()

    def _reset_predictor(self):
        brs_mode = self.state['brs_mode'].get()
        prob_thresh = self.state['prob_thresh'].get()
        net_clicks_limit = None if brs_mode == 'NoBRS' else self.state['predictor_params']['net_clicks_limit'].get()

        self.predictor = get_predictor(self.net, brs_mode, prob_thresh=prob_thresh, zoom_in_params=self.zoomin_params,
                                       predictor_params={
                                           'num_max_points': self.state['predictor_params']['num_max_points'],
                                           'net_clicks_limit': net_clicks_limit,
                                           'max_size': 800
                                       },
                                       brs_opt_func_params={'min_iou_diff': 1e-3},
                                       lbfgs_params={'maxfun': self.state['lbfgs_max_iters'].get()})

        if self.state['_image_nd'] is not None:
            self.predictor.set_input_image(self.state['_image_nd'])

    def _change_brs_mode(self, *args):
        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.set('INF')
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)
        else:
            if self.net_clicks_entry.get() == 'INF':
                self.net_clicks_entry.set(8)
            self.net_clicks_entry.configure(state=tk.NORMAL)
            self.net_clicks_label.configure(state=tk.NORMAL)

        self._reset_predictor()
        assert len(self.state['_pred_probs_history']) == 0

        self.state.update({
            '_zoomin_history': [],
            '_pred_probs_history': [],
        })

        if self.state['_clicker'] is not None:
            self.state['_clicker'].reset_clicks()
        self.state['_clicks_list'] = []

    # ================================================ Canvas callback =================================================
    def _click_callback(self, is_positive, x, y):
        self.canvas.focus_set()

        if self.image_on_canvas is None:
            messagebox.showwarning("Warning", "Please, load an image first")
            return

        if self._check_entry(self):
            click = clicker.Click(is_positive=is_positive, coords=(y, x))

            self._make_prediction_for_one_click(click)
            self._set_click_dependent_widgets_state()
            self._update_image()

    # ================================================= Other routines =================================================
    def _get_current_mask(self):
        prob_thresh = self.state['prob_thresh'].get()

        if len(self.state['_pred_probs_history']) > 0:
            current_mask = (self.state['_pred_probs_history'][-1] > prob_thresh).astype(np.uint8)
            current_mask = current_mask * (self.state['_object_counter'] + 1)
            mask = np.stack(self.state['_object_masks'] + [current_mask], axis=0).max(axis=0)
        elif self.state['_object_counter'] > 0:
            mask = np.stack(self.state['_object_masks'], axis=0).max(axis=0)
        else:
            mask = None

        return mask

    def _update_image(self, reset_canvas=False):
        alpha = self.state['alpha_blend'].get()
        click_radius = self.state['click_radius'].get()

        mask = self._get_current_mask()
        image = vis.draw_with_blend_and_clicks(self.state['_image'], mask=mask, alpha=alpha,
                                               clicks_list=self.state['_clicks_list'], radius=click_radius)

        if self.image_on_canvas is None:
            self.image_on_canvas = CanvasImage(self.canvas_frame, self.canvas)
            self.image_on_canvas.register_click_callback(self._click_callback)
        self.image_on_canvas.reload_image(Image.fromarray(image), reset_canvas)

    def _make_prediction_for_one_click(self, click):
        self.state['_clicks_list'].append(click)
        self.state['_clicker']._add_click(click)

        pred = self.predictor.get_prediction(self.state['_clicker'])
        if self.state['zoomin_params']['use_zoom_in'].get():
            self.state['_zoomin_history'].append({
                '_input_image': self.predictor.zoom_in._input_image,
                '_prev_probs': self.predictor.zoom_in._prev_probs,
                '_object_roi': self.predictor.zoom_in._object_roi,
                '_roi_image': self.predictor.zoom_in._roi_image
            })

        self.state['_pred_probs_history'].append(pred)

    # =========================================== Widget and frame routines ============================================
    def _set_click_dependent_widgets_state(self):
        after_1st_click_state = tk.DISABLED if len(self.state['_clicks_list']) == 0 else tk.NORMAL
        before_1st_click_state = tk.DISABLED if len(self.state['_clicks_list']) > 0 else tk.NORMAL

        self.finish_object_button.configure(state=after_1st_click_state)
        self.undo_click_button.configure(state=after_1st_click_state)
        self.reset_clicks_button.configure(state=after_1st_click_state)
        self.zoomin_options_frame.set_frame_state(before_1st_click_state)
        self.brs_options_frame.set_frame_state(before_1st_click_state)

        if self.state['brs_mode'].get() == 'NoBRS':
            self.net_clicks_entry.configure(state=tk.DISABLED)
            self.net_clicks_label.configure(state=tk.DISABLED)

    def _check_entry(self, widget):
        all_checked = True
        if widget.winfo_children is not None:
            for w in widget.winfo_children():
                all_checked = all_checked and self._check_entry(w)

        if getattr(widget, "_check_bounds", None) is not None:
            all_checked = all_checked and widget._check_bounds(widget.get(), '-1')

        return all_checked
