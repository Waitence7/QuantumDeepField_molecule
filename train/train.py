#!/usr/bin/env python3

import argparse
import os
from pathlib import Path
import pickle
import timeit

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import torch


class QuantumDeepField(nn.Module):
    def __init__(self, device, N_orbitals,
                 dim, layer_functional, operation, N_output,
                 hidden_HK, layer_HK,):
        super(QuantumDeepField, self).__init__()

        """All learning parameters of the QDF model."""
        self.coefficient = nn.Embedding(N_orbitals, dim)
        self.zeta = nn.Embedding(N_orbitals, 1)  # Orbital exponent.
        nn.init.ones_(self.zeta.weight)  # Initialize each zeta with one.
        self.W_functional = nn.ModuleList([nn.Linear(dim, dim)
                                          for _ in range(layer_functional)])
        self.W_property = nn.Linear(dim, N_output)
        self.W_density = nn.Linear(1, hidden_HK)
        self.W_HK = nn.ModuleList([nn.Linear(hidden_HK, hidden_HK)
                                  for _ in range(layer_HK)])
        self.W_potential = nn.Linear(hidden_HK, 1)

        self.device = device
        self.dim = dim
        self.layer_functional = layer_functional
        self.operation = operation
        self.layer_HK = layer_HK

    def list_to_batch(self, xs, dtype=torch.FloatTensor, cat=None, axis=None):
        xs = [dtype(x).to(self.device) for x in xs]
        if cat:
            return torch.cat(xs, axis)
        else:
            return xs

    def pad(self, matrices, pad_value):
        shapes = [m.shape for m in matrices]
        M, N = sum([s[0] for s in shapes]), sum([s[1] for s in shapes])
        pad_matrices = torch.full((M, N), pad_value, device=self.device)
        i, j = 0, 0
        for k, matrix in enumerate(matrices):
            matrix = torch.FloatTensor(matrix).to(self.device)
            m, n = shapes[k]
            pad_matrices[i:i+m, j:j+n] = matrix
            i += m
            j += n
        return pad_matrices

    def basis_matrix(self, atomic_orbitals,
                     distance_matrices, quantum_numbers):
        zetas = torch.squeeze(self.zeta(atomic_orbitals))
        GTOs = (distance_matrices**(quantum_numbers-1) *
                torch.exp(-zetas*distance_matrices**2))
        GTOs = F.normalize(GTOs, 2, 0)
        return GTOs

    def LCAO(self, inputs):
        (atomic_orbitals, distance_matrices,
         quantum_numbers, N_electrons, N_fields) = inputs

        atomic_orbitals = self.list_to_batch(atomic_orbitals, torch.LongTensor)
        distance_matrices = self.pad(distance_matrices, 1e6)
        quantum_numbers = self.list_to_batch(quantum_numbers, cat=True, axis=1)
        N_electrons = self.list_to_batch(N_electrons)

        coefficients = []
        for AOs in atomic_orbitals:
            coefs = F.normalize(self.coefficient(AOs), 2, 0)
            coefficients.append(coefs)
        coefficients = torch.cat(coefficients)
        atomic_orbitals = torch.cat(atomic_orbitals)

        basis_matrix = self.basis_matrix(atomic_orbitals,
                                         distance_matrices, quantum_numbers)
        molecular_orbitals = torch.matmul(basis_matrix, coefficients)

        split_MOs = torch.split(molecular_orbitals, N_fields)
        normalized_MOs = []
        for N_elec, MOs in zip(N_electrons, split_MOs):
            MOs = torch.sqrt(N_elec/self.dim) * F.normalize(MOs, 2, 0)
            normalized_MOs.append(MOs)

        return torch.cat(normalized_MOs)

    def functional(self, vectors, layers, operation, axis):
        for l in range(layers):
            vectors = torch.relu(self.W_functional[l](vectors))
        if operation == 'sum':
            vectors = [torch.sum(vs, 0) for vs in torch.split(vectors, axis)]
        if operation == 'mean':
            vectors = [torch.mean(vs, 0) for vs in torch.split(vectors, axis)]
        return torch.stack(vectors)

    def HKmap(self, scalars, layers):
        vectors = self.W_density(scalars)
        for l in range(layers):
            vectors = torch.relu(self.W_HK[l](vectors))
        return self.W_potential(vectors)

    def forward(self, data, train=False, target=None, predict=False):
        idx, inputs, N_fields = data[0], data[1:6], data[5]

        if predict:
            with torch.no_grad():
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,
                                             self.layer_functional,
                                             self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E_

        elif train:
            molecular_orbitals = self.LCAO(inputs)
            if target == 'E':
                E = self.list_to_batch(data[6], cat=True, axis=0)
                final_layer = self.functional(molecular_orbitals,
                                             self.layer_functional,
                                             self.operation, N_fields)
                E_ = self.W_property(final_layer)
                loss = F.mse_loss(E, E_)
            if target == 'V':
                V = self.list_to_batch(data[7], cat=True, axis=0)
                densities = torch.sum(molecular_orbitals**2, 1)
                densities = torch.unsqueeze(densities, 1)
                V_ = self.HKmap(densities, self.layer_HK)
                loss = F.mse_loss(V, V_)
            return loss

        else:
            with torch.no_grad():
                E = self.list_to_batch(data[6], cat=True, axis=0)
                molecular_orbitals = self.LCAO(inputs)
                final_layer = self.functional(molecular_orbitals,
                                             self.layer_functional,
                                             self.operation, N_fields)
                E_ = self.W_property(final_layer)
                return idx, E, E_


class Trainer(object):
    def __init__(self, model, lr, lr_decay, step_size):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer,
                                                   step_size, lr_decay)

    def optimize(self, loss, optimizer):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def train(self, dataloader):
        losses_E, losses_V = 0, 0
        for data in dataloader:
            loss_E = self.model.forward(data, train=True, target='E')
            self.optimize(loss_E, self.optimizer)
            losses_E += loss_E.item()
            loss_V = self.model.forward(data, train=True, target='V')
            self.optimize(loss_V, self.optimizer)
            losses_V += loss_V.item()
        self.scheduler.step()
        return losses_E, losses_V


class Tester(object):
    def __init__(self, model):
        self.model = model

    def test(self, dataloader, time=False):
        N = sum([len(data[0]) for data in dataloader])
        IDs, Es, Es_ = [], [], []
        SAE = 0
        start = timeit.default_timer()

        for i, data in enumerate(dataloader):
            idx, E, E_ = self.model.forward(data)
            SAE_batch = torch.sum(torch.abs(E - E_), 0)
            SAE += SAE_batch
            IDs += list(idx)
            Es += E.tolist()
            Es_ += E_.tolist()

            if (time is True and i == 0):
                time = timeit.default_timer() - start
                minutes = len(dataloader) * time / 60
                hours = int(minutes / 60)
                minutes = int(minutes - 60 * hours)
                print('The prediction will finish in about',
                      hours, 'hours', minutes, 'minutes.')

        MAE = (SAE/N).tolist()
        MAE = ','.join([str(m) for m in MAE])

        prediction = 'ID\tCorrect\tPredict\tError\n'
        for idx, E, E_ in zip(IDs, Es, Es_):
            error = np.abs(np.array(E) - np.array(E_))
            error = ','.join([str(e) for e in error])
            E = ','.join([str(e) for e in E])
            E_ = ','.join([str(e) for e in E_])
            prediction += '\t'.join([idx, E, E_, error]) + '\n'

        return MAE, prediction

    def save_result(self, result, filename):
        with open(filename, 'a') as f:
            f.write(result + '\n')

    def save_prediction(self, prediction, filename):
        with open(filename, 'w') as f:
            f.write(prediction)

    def save_model(self, model, filename):
        torch.save(model.state_dict(), filename)


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, directory):
        self.directory = os.path.abspath(directory)
        print(f"Directory: {self.directory}")
        paths = sorted(Path(self.directory).iterdir(), key=os.path.getmtime)
        self.files = [p.name for p in paths]
        print(f"Files: {self.files[:5]}")  # First 5 files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.directory, self.files[idx])
        return np.load(file_path, allow_pickle=True)


def collate_fn(xs):
    return list(zip(*xs))


def mydataloader(dataset, batch_size, num_workers, shuffle=False):
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size, shuffle=shuffle, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True)
    return dataloader


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('dataset')
    parser.add_argument('basis_set')
    parser.add_argument('radius')
    parser.add_argument('grid_interval')
    parser.add_argument('dim', type=int)
    parser.add_argument('layer_functional', type=int)
    parser.add_argument('hidden_HK', type=int)
    parser.add_argument('layer_HK', type=int)
    parser.add_argument('operation')
    parser.add_argument('batch_size', type=int)
    parser.add_argument('lr', type=float)
    parser.add_argument('lr_decay', type=float)
    parser.add_argument('step_size', type=int)
    parser.add_argument('iteration', type=int)
    parser.add_argument('setting')
    parser.add_argument('num_workers', type=int)
    args = parser.parse_args()

    dataset = args.dataset
    dataset = 'QM9under7atoms_homolumo_eV'  # Force correct dataset
    print(f"Dataset: {dataset}")
    unit = '(' + dataset.split('_')[-1] + ')'
    basis_set = args.basis_set
    radius = args.radius
    grid_interval = args.grid_interval
    dim = args.dim
    layer_functional = args.layer_functional
    hidden_HK = args.hidden_HK
    layer_HK = args.layer_HK
    operation = args.operation
    batch_size = args.batch_size
    lr = args.lr
    lr_decay = args.lr_decay
    step_size = args.step_size
    iteration = args.iteration
    setting = args.setting
    num_workers = args.num_workers

    torch.manual_seed(1729)

    if torch.xpu.is_available():
        device = torch.device('xpu')
        print('Using Intel NPU (XPU).')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
        print('Using CUDA GPU.')
    else:
        device = torch.device('cpu')
        print('Using CPU.')
    print('-'*50)

    # Dataset directory (absolute path relative to repository root)
    project_root = os.path.dirname(os.path.dirname(__file__))
    dir_dataset = os.path.join(project_root, 'dataset', dataset) + os.sep
    field = '_'.join([basis_set, radius + 'sphere', grid_interval + 'grid/'])
    dataset_train = MyDataset(dir_dataset + 'train_' + field)
    dataset_val = MyDataset(dir_dataset + 'val_' + field)
    dataset_test = MyDataset(dir_dataset + 'test_' + field)
    dataloader_train = mydataloader(dataset_train, batch_size, num_workers, shuffle=True)
    dataloader_val = mydataloader(dataset_val, batch_size, num_workers)
    dataloader_test = mydataloader(dataset_test, batch_size, num_workers)

    print('# of training samples: ', len(dataset_train))
    print('# of validation samples: ', len(dataset_val))
    print('# of test samples: ', len(dataset_test))
    print('-'*50)

    with open(dir_dataset + 'orbitaldict_' + basis_set + '.pickle', 'rb') as f:
        orbital_dict = pickle.load(f)
    N_orbitals = len(orbital_dict)

    N_output = len(dataset_test[0][-2][0])

    print('Set a QDF model.')
    model = QuantumDeepField(device, N_orbitals,
                            dim, layer_functional, operation, N_output,
                            hidden_HK, layer_HK).to(device)

    trainer = Trainer(model, lr, lr_decay, step_size)
    tester = Tester(model)
    print('# of model parameters:',
          sum([np.prod(p.size()) for p in model.parameters()]))
    print('-'*50)

    file_result = '../output/result--' + setting + '.txt'
    result = ('Epoch\tTime(sec)\tLoss_E\tLoss_V\t'
              'MAE_val' + unit + '\tMAE_test' + unit)
    with open(file_result, 'w') as f:
        f.write(result + '\n')
    file_prediction = '../output/prediction--' + setting + '.txt'
    file_model = '../output/model--' + setting

    print('Start training of the QDF model with', dataset, 'dataset.\n'
          'The training result is displayed in this terminal every epoch.\n'
          'The result, prediction, and trained model '
          'are saved in the output directory.\n'
          'Wait for a while...')

    start = timeit.default_timer()

    for epoch in range(iteration):
        loss_E, loss_V = trainer.train(dataloader_train)
        MAE_val = tester.test(dataloader_val)[0]
        MAE_test, prediction = tester.test(dataloader_test)
        time = timeit.default_timer() - start

        if epoch == 0:
            minutes = iteration * time / 60
            hours = int(minutes / 60)
            minutes = int(minutes - 60 * hours)
            print('The training will finish in about',
                  hours, 'hours', minutes, 'minutes.')
            print('-'*50)
            print(result)

        result = '\t'.join(map(str, [epoch, time, loss_E, loss_V,
                                    MAE_val, MAE_test]))
        tester.save_result(result, file_result)
        tester.save_prediction(prediction, file_prediction)
        tester.save_model(model, file_model)
        print(result)

    print('The training has finished.')


