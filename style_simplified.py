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

    def stylize(self, content_image, style_image, *,
        content_weight: float = 0.015,
        style_weight: float = 1.,
        tv_weight: float = 2.,
        max_dim: int = 512,
        iterations: int = 500,
        step_size: float = 0.02,
        avg_decay: float = 0.99,
        init: str = 'content'):

        content_weights = [content_weight / len(self.content_layers)] * len(self.content_layers)

        tv_loss = Scale(LayerApply(TVLoss(),"input"),tv_weight)

        cw, ch = size_to_fit(content_image.size, max_dim, scale_up=True)
        if init == 'content':
            self.image = TF.to_tensor(content_image.resize((cw,ch), Image.Resampling.LANCZOS)).unsqueeze(0)
        else:
            raise Exception(f'Unknown init: {init}')

        self.image = self.image.to("cuda")

        torch.cuda.empty_cache()

        cw, ch = size_to_fit(content_image.size, max_dim, scale_up=True)
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


        sw, sh = size_to_fit(style_image.size, max_dim, scale_up=True)
        style = TF.to_tensor(style_image.resize((sw,sh), Image.Resampling.LANCZOS)).unsqueeze(0).to("cuda")
        print(f"processing style image {sw}x{sh}")
        style_feats = self.model(style, layers=self.style_layers)
        for layer in self.style_layers:
            target = StyleLoss.gram_matrix(style_feats[layer])*style_weight
            style_targets[layer] = target

        for layer, weight in zip(self.style_layers, self.style_layer_weights):
            style_losses.append(Scale(LayerApply(StyleLoss(style_targets[layer]), layer) , weight))

        crit = SumLoss([*content_losses, *style_losses, tv_loss])

        opt = torch.optim.Adam([self.image], lr=step_size)

        torch.cuda.empty_cache()

        for i in range(iterations):
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