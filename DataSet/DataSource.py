import os
import os.path
import torch.utils.data as data
from torchvision import transforms as T
from PIL import Image


class DataSource(data.Dataset):
    def __init__(self, root, train=True, transforms=None):
        self.root = os.path.expanduser(root)  # Rozwijamy ~ do pełnej ścieżki
        self.transforms = transforms          # Transformacje na obrazach (np. resize, normalizacja)
        self.train = train                    # Flaga czy to dane treningowe czy testowe

        self.image_poses = []   # Lista pozycji i orientacji (ground truth)
        self.images_path = []   # Lista ścieżek do obrazów

        self._get_data()       # Wczytanie danych z plików tekstowych

        # Jeśli nie podano transformacji, ustawiamy domyślne
        if transforms is None:
            normalize = T.Normalize(mean=[0.485, 0.456, 0.406],  # Normalizacja jak w ImageNet
                                    std=[0.229, 0.224, 0.225])
            if not train:
                # Transformacje dla danych testowych: resize + center crop + tensor + normalizacja
                self.transforms = T.Compose(
                    [T.Resize(224),
                     T.CenterCrop(224),
                     T.ToTensor(),
                     normalize]
                )
            else:
                # Transformacje dla danych treningowych: resize + losowy crop + tensor + normalizacja
                self.transforms = T.Compose(
                    [T.Resize(256),
                     T.RandomCrop(224),
                     T.ToTensor(),
                     normalize]
                )

    def _get_data(self):
        # Wybór pliku z danymi w zależności od trybu (train/test)
        if self.train:
            txt_file = self.root + 'dataset_train.txt'
        else:
            txt_file = self.root + 'dataset_test.txt'

        # Otwieramy plik i pomijamy 3 pierwsze linie nagłówka
        with open(txt_file, 'r') as f:
            next(f)
            next(f)
            next(f)
            # Dla każdej linii pobieramy nazwę pliku i 7 wartości pozycji + orientacji
            for line in f:
                fname, p0, p1, p2, p3, p4, p5, p6 = line.split()
                # Konwertujemy stringi na floaty
                p0 = float(p0)
                p1 = float(p1)
                p2 = float(p2)
                p3 = float(p3)
                p4 = float(p4)
                p5 = float(p5)
                p6 = float(p6)
                # Dodajemy tuple pozycji i orientacji do listy
                self.image_poses.append((p0, p1, p2, p3, p4, p5, p6))
                # Dodajemy ścieżkę do obrazka do listy
                self.images_path.append(self.root + fname)

    def __getitem__(self, index):
        """
        Zwraca dane dla jednego przykładu:
        - obraz po transformacjach
        - odpowiadające mu pozycję i orientację (7 liczb)
        """
        img_path = self.images_path[index]
        img_pose = self.image_poses[index]
        data = Image.open(img_path)   # Wczytanie obrazu przez PIL
        data = self.transforms(data)  # Transformacje na obrazie
        return data, img_pose

    def __len__(self):
        # Zwraca liczbę przykładów w zbiorze
        return len(self.images_path)
