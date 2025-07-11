from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten
from keras.layers import BatchNormalization,LeakyReLU
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np


def build_genarator():
    noice_shape = (100,)

    model = Sequential([])

    model.add(Dense(256,input_shape=noice_shape))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(104))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))

    model.add(Dense(np.prod((28,28,1)),activation='tanh'))  # 28*28**1
    model.add(Reshape((28,28,1)))  #inverse flattern

    return model


def build_discriminator():

    model = Sequential([])

    model.add(Flatten(input_shape=(28,28,1)))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(1, activation='sigmoid'))

    return model

def train(epoches,batch,save_interval):
    half_batch = int(batch/2)
    (train_data,_),(_,_) = mnist.load_data()
    train_data = (train_data.astype(np.float32) - 127.5) / 127.5 #normalizing -1 +1
    train_data = train_data.reshape(train_data.shape[0],train_data.shape[1],train_data.shape[2],1)
    half_batch = int(batch / 2)
    


    for epoch in range(epoches):
        idx = np.random.randint(0, train_data.shape[0], half_batch)
        imgs = train_data[idx]
 
        noise = np.random.normal(0, 1, (half_batch, 100))   
        gen_imgs = genarator.predict(noise)
        
        discriminator.trainable = True
        
        d_loss_real = discriminator.train_on_batch(imgs, np.ones((half_batch, 1)))
        d_loss_fake = discriminator.train_on_batch(gen_imgs, np.zeros((half_batch, 1)))

        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake) 

        gan_train_noise = np.random.normal(0, 1, (batch, 100)) 
        gan_train_target = np.ones((batch, 1)) #fake but label-1
        
        discriminator.trainable = False
        
        g_loss = GAN.train_on_batch(gan_train_noise, gan_train_target)
        
        print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

        if epoch % save_interval == 0:
            save_imgs(epoch)



def save_imgs(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, 100))
    gen_imgs = genarator.predict(noise)

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
            axs[i,j].axis('off')
            cnt += 1
    
    fig.savefig("images/mnist_%d.png" % epoch,dpi=300)
    plt.close()

optimizer1 = Adam(0.0002, 0.5)
optimizer3 = Adam(0.0002, 0.5)
optimizer2 = Adam(0.0002, 0.5)

genarator = build_genarator()
genarator.compile(optimizer=optimizer2,loss='binary_crossentropy',metrics=['accuracy'])

discriminator = build_discriminator()
discriminator.compile(optimizer=optimizer1,loss='binary_crossentropy',metrics=['accuracy'])

noice_in = Input(shape=(100,))
genarator_out = genarator(noice_in)

discriminator_out = discriminator(genarator_out)

GAN = Model(noice_in,discriminator_out)
GAN.compile(loss="binary_crossentropy",optimizer=optimizer3)

train(10000,64,10)

genarator.save("ganarator.keras")