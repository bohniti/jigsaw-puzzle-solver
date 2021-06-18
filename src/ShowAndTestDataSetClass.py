from SiameseNetworkDataset import SiameseNetworkDataset
import matplotlib.pyplot as plt
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop((378, 371))
])

siamese_dataset = SiameseNetworkDataset(
    csv_file='/Users/beantown/PycharmProjects/jigsaw-puzzle-solver/data/hisfrag20/prepared/paris_as_csv/train.csv', transform=transform)


fig = plt.figure()
ax1 = fig.add_subplot(2,2,1)
ax1.imshow(siamese_dataset[0]['img_0'].permute(1, 2, 0))
ax2 = fig.add_subplot(2,2,2)
ax2.imshow(siamese_dataset[0]['img_1'].permute(1, 2, 0))
ax3 = fig.add_subplot(2,2,3)
ax3.imshow(siamese_dataset[100]['img_0'].permute(1, 2, 0))
ax4 = fig.add_subplot(2,2,4)
ax4.imshow(siamese_dataset[100]['img_1'].permute(1, 2, 0))


plt.show()