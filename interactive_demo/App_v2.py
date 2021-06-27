from interactive_demo import controller_v2
from isegm.inference import clicker

class AppReplacement:

    def __init__(self, image, threshold, model, xcoords, ycoords, pos, device, limit_longest_size, predictor_params):

        self.net = model.to(device)
        self.device = device
        self.predictor_params = predictor_params
        self.filename = ''
        self.filenames = []
        self.current_file_index = 0
        self.xcoords = xcoords
        self.ycoords = ycoords
        self.pos = pos
        self.image = image
        self._result_mask = None
        self.result_mask = None
        self.probs_history = []
        self.clicker = clicker.Clicker()
        self.predictor = None
        self.states = []
        self.threshold = threshold
        self.original_image = None
        self.prediction = None
        self.controller = controller_v2.InteractiveController_v2(model, xcoords, ycoords, pos, device,
                                                                 predictor_params=predictor_params,
                                                                 update_image_callback=False)
        self.brs_modes = ['NoBRS', 'RGB-BRS', 'DistMap-BRS', 'f-BRS-A', 'f-BRS-B', 'f-BRS-C']
        self.limit_longest_size = limit_longest_size

        self.controller.set_image(image)
        self.prediction = self.controller.prediction

