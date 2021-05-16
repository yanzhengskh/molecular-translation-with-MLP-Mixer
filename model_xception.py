import torch
import torch.nn as nn

class SeperableConv2d(nn.Module):

    #***Figure 4. An “extreme” version of our Inception module,
    #with one spatial convolution per output channel of the 1x1
    #convolution."""
    def __init__(self, input_channels, output_channels, kernel_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels,
            kernel_size,
            groups=input_channels,
            bias=False,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)

        return x

class LinearConv2d(nn.Module):

    def __init__(self, input_channels, output_channels, input_img_size, output_img_size, **kwargs):

        super().__init__()
        self.depthwise = nn.Conv2d(
            input_channels,
            input_channels * output_img_size[0] * output_img_size[1],
            input_img_size,
            groups=input_channels,
            bias=True,
            **kwargs
        )

        self.pointwise = nn.Conv2d(input_channels, output_channels, 1, bias=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = x.reshape(x.shape[0], input_channels, output_img_size[0], output_img_size[1])
        x = self.pointwise(x)

        return x

class EntryFlow(nn.Module):

    def __init__(self):

        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.conv3_residual = nn.Sequential(
            SeperableConv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1),
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2),
            nn.BatchNorm2d(128),
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 256, 1, stride=2),
            nn.BatchNorm2d(256),
        )

        #no downsampling
        self.conv5_residual = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(256, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.MaxPool2d(3, 1, padding=1)
        )

        #no downsampling
        self.conv5_shortcut = nn.Sequential(
            nn.Conv2d(256, 728, 1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        residual = self.conv3_residual(x)
        shortcut = self.conv3_shortcut(x)
        x = residual + shortcut
        residual = self.conv4_residual(x)
        shortcut = self.conv4_shortcut(x)
        x = residual + shortcut
        residual = self.conv5_residual(x)
        shortcut = self.conv5_shortcut(x)
        x = residual + shortcut

        return x

class MiddleFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual

class LinearFLowBlock(nn.Module):

    def __init__(self):
        super().__init__()

        self.shortcut = nn.Sequential()
        self.conv1 = nn.Sequential(
            nn.ReLU(inplace=True),
            LinearConv2d(512, 512, (20, 40), (20, 40), padding=0),
            nn.BatchNorm2d(512)
        )
        self.conv2 = nn.Sequential(
            nn.ReLU(inplace=True),
            LinearConv2d(512, 512, (20, 40), (20, 40), padding=0),
            nn.BatchNorm2d(512)
        )
        self.conv3 = nn.Sequential(
            nn.ReLU(inplace=True),
            LinearConv2d(512, 512, (20, 40), (20, 40), padding=0),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.conv2(residual)
        residual = self.conv3(residual)

        shortcut = self.shortcut(x)

        return shortcut + residual

class MiddleFlow(nn.Module):
    def __init__(self, block):
        super().__init__()

        #"""then through the middle flow which is repeated eight times"""
        self.middel_block = self._make_flow(block, 8)

    def forward(self, x):
        x = self.middel_block(x)
        return x

    def _make_flow(self, block, times):
        flows = []
        for i in range(times):
            flows.append(block())

        return nn.Sequential(*flows)


class ExitFLow(nn.Module):

    def __init__(self):
        super().__init__()
        self.residual = nn.Sequential(
            nn.ReLU(),
            SeperableConv2d(728, 728, 3, padding=1),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeperableConv2d(728, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.shortcut = nn.Sequential(
            nn.Conv2d(728, 512, 1, stride=2),
            nn.BatchNorm2d(512)
        )

    def forward(self, x):
        shortcut = self.shortcut(x)
        residual = self.residual(x)
        output = shortcut + residual
        return output


class MeiYun(pl.LightningModule):

    def __init__(self, config = None):
        super().__init__()
        self.save_hyperparameters(config)
        self.len_vocab = len(VOCAB)
        self.entry_flow = EntryFlow()
        self.middel_flow = MiddleFlow(MiddleFLowBlock)
        self.exit_flow = ExitFLow()
        self.linear_flow = MiddleFlow(LinearFLowBlock)
        self.linear_conv = nn.Sequential(
            LinearConv2d(512, 400, (20, 40), padding=0),
            nn.BatchNorm2d(400),
            nn.ReLU(inplace=True),
            LinearConv2d(400, self.hparams.max_len, (20, 40), padding=0),
            nn.BatchNorm2d(self.hparams.max_len),
            nn.ReLU(inplace=True)
        )
        self.final_conv = nn.Conv2d(
            self.hparams.max_len,
            self.hparams.max_len * self.len_vocab,
            (20, 40),
            groups=self.hparams.max_len,
            bias=True,
            padding=0
        )

        self.apply(self._init_weights)

    def forward(self, x):
        x = self.entry_flow(x)
        x = self.middel_flow(x)
        x = self.exit_flow(x)
        x = self.linear_flow(x)
        x = self.linear_conv(x)
        x = self.final_conv(x)
        x = x.reshape(x.shape[0], self.hparams.max_len, self.len_vocab)

        return x

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, (nn.LayerNorm, nn.BatchNorm2d)):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def predit(self, x, EOS=2, temp=1.):
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            B = x.shape[0]
            # start of sentence
            trg_input = torch.tensor([], dtype=torch.long, device=self.device).expand(B, 0)
            logits = self(images) / temp
            probs = F.softmax(logits, dim=-1)
            pred = torch.argmax(probs, -1, keepdim=True)
            i = 0
            while True:
                trg_input = torch.cat([trg_input, pred[:, i*self.len_vocab: (i+1)*self.len_vocab]], 1)
                i += 1
                if torch.any(trg_input == EOS, 1).sum().item() == B or trg_input.shape[1] >= self.hparams.max_len:
                    return trg_input

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.transpose(1,2), y[:,1:]) 
        self.log('loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat.transpose(1,2), y[:,1:]) 
        self.log('val_loss', loss, prog_bar=True)
    
    def configure_optimizers(self):
        optimizer = getattr(torch.optim, self.hparams.optimizer)(self.parameters(), lr=self.hparams.lr)
        if 'scheduler' in self.hparams:
            schedulers = [
                getattr(torch.optim.lr_scheduler, scheduler)(optimizer, **params)
                for scheduler, params in self.hparams.scheduler.items()
            ]
            return [optimizer], schedulers 
        return optimizer