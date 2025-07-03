import torch
import time
import numpy as np
import torch.utils.data as Data
import torch.nn as nn
from torch.autograd import Variable
import PoseNet.PoseNet as PoseNet
from DataSet import *

learning_rate = 0.0001
batch_size = 75
EPOCH = 1000
directory = 'KingsCollege/KingsCollege/'

# Wybór urządzenia: GPU jeśli jest dostępne, inaczej CPU
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def main():
    print(torch.cuda.get_device_name(device))  # Nazwa GPU (jeśli jest)

    datasource = DataSource(directory, train=True)
    train_loader = Data.DataLoader(dataset=datasource, batch_size=batch_size, shuffle=True)

    posenet = PoseNet.posenet_v1().to(device)

    criterion = PoseNet.PoseLoss(0.3, 150, 0.3, 150, 1, 500)

    optimizer = torch.optim.SGD(nn.ParameterList(posenet.parameters()), lr=learning_rate)

    start_time = time.time()  

    best_loss = float('inf')  # Do przechowywania najlepszego wyniku (najniższej straty)

    for epoch in range(EPOCH):
        running_loss = 0.0
        for step, (images, poses) in enumerate(train_loader):
            b_images = Variable(images, requires_grad=True).to(device)

            poses = torch.stack(poses).transpose(0, 1).float().to(device)
            b_poses = poses.clone().detach().requires_grad_(True)

            p1_x, p1_q, p2_x, p2_q, p3_x, p3_q = posenet(b_images)
            loss = criterion(p1_x, p1_q, p2_x, p2_q, p3_x, p3_q, b_poses)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if step % 20 == 0:
                elapsed = time.time() - start_time
                avg_epoch_time = elapsed / (epoch + 1)
                eta = avg_epoch_time * (EPOCH - epoch - 1)
                print(f"Iteration (Epoch): {epoch}, Step: {step}")
                print(f"    Loss is: {loss.item():.6f}")
                print(f"    Total time elapsed: {elapsed:.2f}s")
                print(f"    ETA: {eta / 60:.2f} minutes remaining")

        # Średnia strata w danym epochu
        epoch_loss = running_loss / len(train_loader)

        # Sprawdzamy, czy aktualna strata jest lepsza niż dotychczas najlepsza
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(posenet.state_dict(), 'best_posenet_model.pth')

        # Co 100 epok zapisujemy model (nawet jeśli nie jest najlepszy)
        if (epoch + 1) % 100 == 0:
            checkpoint_path = f'ModelCheckpoints/posenet_checkpoint_epoch_{epoch+1}.pth'
            torch.save(posenet.state_dict(), checkpoint_path)


if __name__ == '__main__':
    main()
