from stylemodmod import *

class StyleTransfer:
    def __init__(self, pooling="max"):
        self.image = None
        self.average = None

        self.content_layers = [22]
        self.style_layers = [1, 6, 11, 20, 29]

        style_weights = [256,64,16,4,1]
        weight_sum = sum(style_weights)
        self.style_layer_weights = [w/weight_sum for w in style_weights]

        self.model = VGGFeatures(self.style_layers + self.content_layers, pooling=pooling).to("cuda")

    def stylize(self, content_image, style_images, *,
        style_weights=None,
        content_weight: float = 0.015,
        tv_weight: float = 2.,
        min_scale: int = 128,
        end_scale: int = 512,
        iterations: int = 500,
        initial_iterations: int = 1000,
        step_size: float = 0.02,
        avg_decay: float = 0.99,
        init: str = 'content',
        style_scale_fac: float = 1.,
        style_size: int = None,
        callback=None):

        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)
        
        #note, this is the weigth of the style images if there are more than one, 
        # different the the weight of the style layers
        if style_weights is None:
            style_weights = [1 / len(style_images)] * len(style_images)
        else:
            weight_sum = sum(abs(w) for w in style_weights)
            style_weights = [weight / weight_sum for weight in style_weights]

        tv_loss = Scale(LayerApply(TVLoss(),"input"),tv_weight)

        scales = gen_scales(min_scale, end_scale)

        cw, ch = size_to_fit(content_image.size, scales[0], scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw,ch), Image.Resampling.LANCZOS)).unsqueeze(0)
        else:
            raise Exception(f'Unknown init: {init}')

        self.image = self.image.to("cuda")

        #opt = None

        for scale in scales:
            torch.cuda.empty_cache()

            cw, ch = size_to_fit(content_image.size, scale, scale_up=True)
            content = TF.to_tensor(content_image.resize((cw,ch), Image.Resampling.LANCZOS)).unsqueeze(0).to("cuda")

            #why do we need to interpolate the content image?
            self.image = F.interpolate(self.image.detach(), size=(ch,cw), mode='bicubic').clamp(0,1)

            self.average = EMA(self.image, avg_decay)

            self.image.requires_grad_()

            print(f"processing content image {cw}x{ch}")
            content_feats = self.model(content, layers=self.content_layers)
            content_losses = []
            for layer, weight in zip(self.content_layers, content_weights):
                content_losses.append(Scale(LayerApply(ContentLoss(content_feats[layer]), layer) , weight))

            style_targets = {}
            style_losses = []

            for i, image in enumerate(style_images):
                if style_size is None:
                    sw, sh = size_to_fit(image.size, round(scale*style_scale_fac))
                else:
                    sw, sh = size_to_fit(image.size, style_size)
                style = TF.to_tensor(image.resize((sw,sh), Image.Resampling.LANCZOS)).unsqueeze(0).to("cuda")
                print(f"processing style image {i} {sw}x{sh}")
                style_feats = self.model(style, layers=self.style_layers)
                for layer in self.style_layers:
                    target = StyleLoss.gram_matrix(style_feats[layer])*style_weights[i]
                    #what, are we just taking weighted average for the different style images?
                    #don't want to compute the loss seprately for each style image?
                    if layer not in style_targets:
                        style_targets[layer] = target
                    else:
                        style_targets[layer] += target

            for layer, weight in zip(self.style_layers, self.style_layer_weights):
                style_losses.append(Scale(LayerApply(StyleLoss(style_targets[layer]), layer) , weight))

            crit = SumLoss([*content_losses, *style_losses, tv_loss])

            opt = torch.optim.Adam([self.image], lr=step_size)

            torch.cuda.empty_cache()

            actual_its = initial_iterations if scale == scales[0] else iterations

            for i in range(1, actual_its+1):
                feats = self.model(self.image)
                loss = crit(feats)
                opt.zero_grad()
                loss.backward()
                opt.step()

                with torch.no_grad():
                    self.image.clamp(0,1)
                self.average.update(self.image)

                with torch.no_grad():
                    self.image.copy_(self.average.get())

        return self.get_image()

    def get_image_tensor(self):
        return self.average.get().detach()[0].clamp(0, 1)

    def get_image(self, image_type='pil'):
        if self.average is not None:
            image = self.get_image_tensor()
            if image_type.lower() == 'pil':
                return TF.to_pil_image(image)
            elif image_type.lower() == 'np_uint16':
                arr = image.cpu().movedim(0, 2).numpy()
                return np.uint16(np.round(arr * 65535))
            else:
                raise ValueError("image_type must be 'pil' or 'np_uint16'")