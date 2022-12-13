import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt


class Hopfield:
    def __init__(self, input_shape):
        self.train_data = []

        # init weights as zeros
        self.W = np.zeros([input_shape[0]*input_shape[1],
                          input_shape[0]*input_shape[1]], dtype=np.int8)

    # calculates energy
    def energy(self, o):
        e = -0.5*np.matmul(np.matmul(o.T, self.W), o)
        return e

    def add_train(self, img_path):
        img = plt.imread(img_path)
        img = np.mean(img, axis=2)
        if (img.shape != (32, 32)):
            print('Error: image shape ', img.shape, ' is not (32,32)')
            return

        img_mean = np.mean(img)
        img = np.where(img < img_mean, -1, 1)
        train_data = img.flatten()

        # rearrange flatten image to matrix
        for i in range(train_data.size):
            for j in range(i, train_data.size):
                if i == j:
                    self.W[i][j] = 0
                else:
                    w_ij = train_data[i] * train_data[j]
                    self.W[i][j] += w_ij
                    self.W[j][i] += w_ij

    # function that updates neuron
    def update(self, state, idx=None):
        if (idx == None):
            new_state = np.matmul(self.W, state)
            new_state[new_state < 0] = -1
            new_state[new_state > 0] = 1
            new_state[new_state == 0] = state[new_state == 0]
            state = new_state
        else:
            new_state = np.matmul(self.W[idx], state)
            if new_state < 0:
                state[idx] = -1
            else:
                state[idx] = 1
        return state

    # if we set async parameter to 1024, then 1 iteration is enough
    def predict(self, mat_input, iteration, async_iteration=200):
        input_shape = mat_input.shape
        fig, axs = plt.subplots(1, 1)
        axs.axis('off')
        print(input_shape)
        graph = axs.imshow(mat_input*255, cmap='binary')
        mat_input = np.where(mat_input < 0.5, -1, 1)
        fig.canvas.draw_idle()
        plt.pause(1)

        e_list = []

        e = self.energy(mat_input.flatten())
        e_list.append(e)
        state = mat_input.flatten()

        for i in range(iteration):
            for j in range(async_iteration):
                # print('Async weights updates, iteration#', j)
                idx = np.random.randint(state.size)
                state = self.update(state, idx)

            state_show = np.where(state < 1, 0, 1).reshape(inpuinput_shapet_shape)
            graph.set_data(state_show*255)
            axs.set_title('Async update Iteration #%i' % i)
            fig.canvas.draw_idle()
            plt.pause(0.25)
            new_e = -0.5*np.matmul(np.matmul(state.T, self.W), state)
            print('Iteration#', i, ', Energy: ', new_e)
            # there are cases where energy remains the same, but image is not restored
            # if new_e == e:
            #     print('Energy is the same.')
            #     break
            e = new_e
            e_list.append(e)

        return np.where(state < 1, 0, 1).reshape(input_shape), e_list


def getOptions():
    parser = argparse.ArgumentParser(description='parses command.')
    parser.add_argument('-t', '--train', nargs='*',
                        help='training data.')
    parser.add_argument('-i', '--iteration', type=int,
                        help='number of iteration.')
    options = parser.parse_args(sys.argv[1:])
    return options


if __name__ == '__main__':
    np.random.seed(1)
    options = getOptions()
    input_shape = (32, 32)
    model = Hopfield(input_shape)
    print('Model initialized with weights shape ', model.W.shape)

    for train_data in options.train:
        print('Start training ', train_data, '...')
        model.add_train(train_data)
        print(train_data, 'training completed!')

    mat_input = np.mean(plt.imread(
        options.train[0]), axis=2) + np.random.uniform(-1, 1, input_shape)
    mat_input = np.where(mat_input < 0.5, 0, 1)

    output_async, e_list_async = model.predict(
        mat_input, options.iteration)

    fig = plt.figure(1)
    fig.suptitle('Result with %i iteration' % options.iteration)
    axs1 = fig.add_subplot(231)
    axs1.set_title('Input')
    axs1.imshow(mat_input*255, cmap='binary')
    axs2 = fig.add_subplot(232)
    axs2.set_title('Async Update')
    axs2.imshow(output_async*255, cmap='binary')

    input("Press Enter to continue...")
