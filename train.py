import argparse
import time
from sys import platform
from pathlib import Path

from models import *
from utils.datasets import *
from utils.utils import *

# batch_size 8: 32*17 = 544
# batch_size 4: 32*25 = 800 (1.47 vs 544) or 32*23 = 736
# batch_size 2: 32*35 = 1120 (1.40 vs 800, 2.06 cumulative)
# batch_size 1: 32*49 = 1568 (1.40 vs 1120, 2.88 cumulative)

parser = argparse.ArgumentParser()
parser.add_argument('--epochs', type=int, default=999, help='number of epochs')
parser.add_argument('--checkpoint', type=int, default=100, help='checkpoint weights are saved after every n epochs')
parser.add_argument('--batch-size', type=int, default=8, help='size of each image batch')
parser.add_argument('--cfg', type=str, default='cfg/c60_a30symmetric.cfg', help='cfg file path')
parser.add_argument('--classes', type=int, default=60, help='number of classes for training')
parser.add_argument('--weights', type=str, default='weights/latest.pt', help='initial weights file path')
parser.add_argument('--name', type=str, default='', help='name of output weights file')
parser.add_argument('--targets', type=str, default='utils/targets_c60.mat', help='targets mat file')
parser.add_argument('--img-size', type=int, default=32 * 25, help='size of each image dimension')
parser.add_argument('--resume', default=False, help='resume training flag')
opt = parser.parse_args()
print(opt)
if opt.name:
    opt.name = '_' + opt.name


def main(opt):
    os.makedirs('weights', exist_ok=True)
    cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if cuda:
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = True

    # Initialize model
    model = Darknet(opt.cfg, opt.classes, opt.img_size)

    # Get dataloader
    dataloader = ListDataset(batch_size=opt.batch_size, img_size=opt.img_size, targets_path=opt.targets)

    # reload saved optimizer state
    
    if opt.resume:
        checkpoint = torch.load(opt.weights, map_location='cpu')

        model.load_state_dict(checkpoint['model'])
        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()

        # # Transfer learning
        # for i, (name, p) in enumerate(model.named_parameters()):
        #     #name = name.replace('module_list.', '')
        #     #print('%4g %70s %9s %12g %20s %12g %12g' % (
        #     #    i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
        #     if p.shape[0] != 650:  # not YOLO layer
        #         p.requires_grad = False

        # Set optimizer
        # optimizer = torch.optim.SGD(model.parameters(), lr=.001, momentum=.9, weight_decay=0.0005 * 0, nesterov=True)
        # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        best_loss = checkpoint['best_loss']

        del checkpoint  # current, saved
    else:
        start_epoch = 0
        best_loss = float('inf')

        if torch.cuda.device_count() > 1:
            print('Using ', torch.cuda.device_count(), ' GPUs')
            model = nn.DataParallel(model)
        model.to(device).train()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4, weight_decay=5e-4)

    # Set scheduler
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 24, eta_min=0.00001, last_epoch=-1)
    # y = 0.001 * exp(-0.00921 * x)  # 1e-4 @ 250, 1e-5 @ 500
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99082, last_epoch=start_epoch - 1)

    modelinfo(model)
    t0, t1 = time.time(), time.time()
    print('%10s' * 16 % (
        'Epoch', 'Batch', 'x', 'y', 'w', 'h', 'conf', 'cls', 'total', 'P', 'R', 'nGT', 'TP', 'FP', 'FN', 'time'))
    # class_weights = xview_class_weights(range(opt.classes)).to(device)
    class_weights_gpu = dataloader.class_weights.to(device)
    for epoch in range(opt.epochs):
        epoch += start_epoch

        # img_size = random.choice([19, 20, 21, 22, 23, 24, 25]) * 32
        # dataloader = ListDataset(train_path, batch_size=opt.batch_size, img_size=img_size, targets_path=opt.targets)
        # print('Running image size %g' % img_size)

        # Update scheduler
        # if epoch % 25 == 0:
        #     scheduler.last_epoch = -1  # for cosine annealing, restart every 25 epochs
        # scheduler.step() 
        # if epoch <= 100:
        # for g in optimizer.param_groups:
        # g['lr'] = 0.0005 * (0.992 ** epoch)  # 1/10 th every 250 epochs
        # g['lr'] = 0.001 * (0.9773 ** epoch)  # 1/10 th every 100 epochs
        # g['lr'] = 0.0005 * (0.955 ** epoch)  # 1/10 th every 50 epochs
        # g['lr'] = 0.0005 * (0.926 ** epoch)  # 1/10 th every 30 epochs

        ui = 0
        rloss = defaultdict(float)  # running loss
        metrics = torch.zeros(4, opt.classes)
        for i, (imgs, targets) in enumerate(dataloader):

            batch_size = 4  # number of pictures at a time
            for j in range(int(len(imgs) / batch_size)):
                targets_j = targets[j * batch_size:j * batch_size + batch_size]
                nGT = sum([len(x) for x in targets_j])
                if nGT < 1:
                    continue

                loss = model(imgs[j * batch_size:j * batch_size + batch_size].to(device), targets_j, requestPrecision=True,
                             weight=class_weights_gpu, epoch=epoch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                metrics += model.losses['metrics']
                for key, val in model.losses.items():
                    rloss[key] = (rloss[key] * ui + val) / (ui + 1)
                ui += 1

                # Precision
                precision = metrics[0] / (metrics[0] + metrics[1] + 1e-16)
                k = (metrics[0] + metrics[1]) > 0
                if k.sum() > 0:
                    mean_precision = precision[k].mean()
                else:
                    mean_precision = 0

                # Recall
                recall = metrics[0] / (metrics[0] + metrics[2] + 1e-16)
                k = (metrics[0] + metrics[2]) > 0
                if k.sum() > 0:
                    mean_recall = recall[k].mean()
                else:
                    mean_recall = 0

                s = ('%10s%10s' + '%10.3g' * 14) % (
                    '%g/%g' % (epoch, opt.epochs - 1), '%g/%g' % (i, len(dataloader) - 1), rloss['x'],
                    rloss['y'], rloss['w'], rloss['h'], rloss['conf'], rloss['cls'],
                    rloss['loss'], mean_precision, mean_recall, model.losses['nGT'], model.losses['TP'],
                    model.losses['FP'], model.losses['FN'], time.time() - t1)
                t1 = time.time()
                print(s)


        # Write epoch results
        with open('results' + opt.name + '.txt', 'a') as file:
            file.write(s + '\n')

        # Update best loss
        loss_per_target = rloss['loss'] / rloss['nGT']
        if loss_per_target < best_loss:
            best_loss = loss_per_target

        # Save latest checkpoint
        checkpoint = {'epoch': epoch,
                      'best_loss': best_loss,
                      'model': model.state_dict(),
                      'optimizer': optimizer.state_dict()}
        torch.save(checkpoint, 'weights' + os.sep + 'latest' + opt.name + '.pt')

        # Save best checkpoint
        if best_loss == loss_per_target:
            os.system('cp weights' + os.sep + 'latest' + opt.name + '.pt weights' + os.sep + 'best' + opt.name + '.pt')

        # Save backup checkpoint
        if (epoch > 0) & (epoch % opt.checkpoint == 0):
            os.system('cp weights' + os.sep + 'latest' + opt.name + '.pt weights' + os.sep + 'backup' + opt.name + '_' + str(epoch) + '.pt')

    # Save final model
    dt = time.time() - t0
    print('Finished %g epochs in %.2fs (%.2fs/epoch)' % (epoch, dt, dt / (epoch + 1)))


if __name__ == '__main__':
    torch.cuda.empty_cache()
    main(opt)
    torch.cuda.empty_cache()
