
### Buoc 1: Load du lieu
# Vi dataset la cac file duoi cham p nen can phai dung pickle de khu tuan tu hoa.
# Load pickled data
import pickle

#load datasets

training_file = 'data/train.p'
validation_file='data/valid.p'
testing_file = 'data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(validation_file, mode='rb') as f:
    valid = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_valid, y_valid = valid['features'], valid['labels']
X_test, y_test = test['features'], test['labels']


### Buoc 2: Gian luoc va khao sat du lieu
# 
# Du lieu duoc chon gom co 4 tu khoa tuong ung voi 4 gia tri:
# 
# -'features' la mot mang 4 chieu chua du lieu pixel tho cua bien bao(so luong,chieu rong,chieu cao,kenh). 
# -'labels' la mang 1 chieu chua cac lop cua bien bao giao thong, duoc anh xa trong tep signnames.csv.
# -'sizes' la mot danh sach chua cac bo du lieu(chieu rong,chieu cao)cua cac anh ban dau.
# -'coords' la mot danh sach chua cac bo du lieu toa do(x1,y1,x2,y2) cua bien bao trong hinh anh. Tu do de cat ra mot buc anh 32x32 chi chua bien bao.
# 
# Dung thu vien numpy de gian luoc du lieu

# So luong du lieu huan luyen
n_train = len(y_train)

# So luong gia tri cong nhan
n_validation = len(y_valid)

# So luong gia tri kiem tra
n_test = len(y_test)

# Hinh dang du lieu hinh anh
image_shape = X_train[0].shape[:-1]

# So lop bien bao co trong du lieu kiem tra
n_classes = len(set(y_test))

print("Number of training examples =", n_train)
print("Number of validation examples =", n_validation)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)


# Khao sat bo du lieu dataset voi bieu do histogram dua vao so lieu trong file signnames.csv

# Load file csv
import  csv
labels  = []
with open('signnames.csv', 'r') as f:
    reader = csv.reader(f)
    for label in reader:
        labels.append(label[1])
labels = labels[1:]

print(labels[0:5],'...')

# Hien thi anh cua 43 loai bien bao 
import matplotlib.pyplot as plt
import random

rows = 7
cols = 7
fig, axs = plt.subplots(rows,cols)
fig.set_size_inches(25,25)

for i in range(43):

    class_image = X_train[y_train==i]
    index = random.randint(0, len(class_image)-1)
    image = class_image[index]

    axs[int(i/rows)][i%cols].imshow(image)
    axs[int(i/rows)][i%cols].set_title(labels[i], fontsize=11)

plt.show()


# Ve bieu do histogram the hien su phan bo cua tap du lieu dataset

# Moi cot trong bieu do, bieu thi cho moi lop trong tap du lieu
n_bins = n_classes

# Ve mot subplot 1x3 de ve 3 histogram
fig, axs = plt.subplots(1, 3, sharey=False, tight_layout=True, figsize=[15,5])

# Dat tieu cho cac histogram va gan cac gan cac gia tri tuong ung.
axs[0].set_title('Train classes distribution histogram', fontsize=11)
axs[1].set_title('Validation classes distribution histogram', fontsize=11)
axs[2].set_title('Test classes distribution histogram', fontsize=11)

x_train_hist, y_train_hist, _ = axs[0].hist(y_train, bins=n_bins)
x_valid_hist, y_valid_hist, _ = axs[1].hist(y_valid, bins=n_bins)
x_test_hist, y_test_hist, _ = axs[2].hist(y_test, bins=n_bins)

print("Maximum class labels instances in train data",x_train_hist.max())
print("Minimum class labels instances in test data",x_train_hist.min(),'\n')

print("Maximum class labels instances in validation data",x_valid_hist.max())
print("Mininum class labels instances in validation data",x_valid_hist.min(),'\n')

print("Maximum class labels instances in test data",x_test_hist.max())
print("Minimum class labels instances in test data",x_test_hist.min())



### Buoc 3: Thiet ke va huan luyen mo hinh

# Tien xu ly du lieu 
# Du lieu duoc dua ve 2 gia tri 0 va 1
def preprocess_dataset(dataset):

    # normalize dataset
    dataset = (dataset.astype(float))/float(255)

    return dataset


# Tang du lieu
# Muc dich cua viec sao chep cac class label hiep gap trong tap du lieu nham giam phuong sai cao trong mo hinh de cho do chinh xac cao hon.
# Nhung no lai lam cham qua trinh training nen buoc phai comment buoc nay lai.
from utils import *

def augment_image(img):    
    img = clipped_zoom(img)
    img = sharpen_image(img)
    img = contr_img(img)
    img = translate_image(img)
    #img = rotate_images(img)  
    #img = add_salt_pepper_noise(img)
    
    return img


X_train= preprocess_dataset(X_train)
X_test= preprocess_dataset(X_test)
X_valid= preprocess_dataset(X_valid)

# np.save('X_train',X_process)
# np.save('X_test',X_valid)
# np.save('X_valid',X_test)


# Vo hieu hoa cac canh bao cua phien ban tensorflow cu. 
import logging
class WarningFilter(logging.Filter):
    def filter(self, record):
        msg = record.getMessage()
        tf_warning = 'retry (from tensorflow.contrib.learn.python.learn.datasets.base)' in msg
        return not tf_warning
            
logger = logging.getLogger('tensorflow')
logger.addFilter(WarningFilter())


# Kien truc mo hinh

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.framework as ops
from sklearn.utils import shuffle


def get_inception_layer( inputs, conv11_size, conv33_11_size, conv33_size,
                         conv55_11_size, conv55_size, pool11_size):
    with tf.variable_scope("conv_1x1"):
        conv11 = layers.conv2d( inputs, conv11_size, [ 1, 1 ] )
    with tf.variable_scope("conv_3x3"):
        conv33_11 = layers.conv2d( inputs, conv33_11_size, [ 1, 1 ] )
        conv33 = layers.conv2d( conv33_11, conv33_size, [ 3, 3 ] )
    with tf.variable_scope("conv_5x5"):
        conv55_11 = layers.conv2d( inputs, conv55_11_size, [ 1, 1 ] )
        conv55 = layers.conv2d( conv55_11, conv55_size, [ 5, 5 ] )
    with tf.variable_scope("pool_proj"):
        pool_proj = layers.max_pool2d( inputs, [ 3, 3 ], stride = 1 )
        pool11 = layers.conv2d( pool_proj, pool11_size, [ 1, 1 ] )
    if tf.__version__ == '0.11.0rc0':
        return tf.concat(3, [conv11, conv33, conv55, pool11])
    return tf.concat([conv11, conv33, conv55, pool11], 3)

end_points = {}

def model(inputs, dropout_keep_prob=0.5, num_classes=43, is_training=True, scope=''):     
    with tf.name_scope(scope, "model", [inputs] ):
        with ops.arg_scope( [ layers.max_pool2d ], padding = 'SAME'):
            end_points['conv0'] = layers.conv2d( inputs, 64, [ 7, 7 ], stride = 2, scope = 'conv0')

            with tf.variable_scope("inception_3a"):
                end_points['inception_3a'] = get_inception_layer( end_points['conv0'], 64, 96, 128, 16, 32, 32)

            with tf.variable_scope("inception_3b"):
                end_points['inception_3b'] = get_inception_layer( end_points['inception_3a'], 128, 128, 192, 32, 96, 64)

            end_points['pool2'] = layers.max_pool2d(end_points['inception_3b'], [ 3, 3 ], scope='pool2')

            #print(end_points['pool2'].shape)

            end_points['reshape'] = tf.reshape( end_points['pool2'], [-1, 8*8*480] )

            end_points['fully_2'] = layers.fully_connected(  end_points['reshape'], 200, activation_fn=tf.nn.relu, scope='fully_2')
            end_points['dropout1'] = layers.dropout( end_points['fully_2'], dropout_keep_prob, is_training = is_training )

            end_points['fully_3'] = layers.fully_connected(  end_points['dropout1'], 400, activation_fn=tf.nn.relu, scope='fully_3')
            end_points['dropout2'] =layers.dropout( end_points['fully_3'], dropout_keep_prob, is_training = is_training)

            end_points['fully_4'] = layers.fully_connected(  end_points['dropout2'], 300, activation_fn=tf.nn.relu, scope='fully_4')
            end_points['dropout3'] = layers.dropout( end_points['fully_4'], dropout_keep_prob, is_training = is_training )

            end_points['logits'] = layers.fully_connected( end_points['dropout3'], num_classes, activation_fn=None, scope='logits')
            end_points['predictions'] = tf.nn.softmax(end_points['logits'], name='predictions')

    return end_points['logits'], end_points


## Train, Validate va Test mo hinh

# Cac thong so cua mo hinh

EPOCHS = 350
BATCH_SIZE = 128

tf.reset_default_graph()

x = tf.placeholder(tf.float32, (None, 32, 32, 3), name='X')
y = tf.placeholder(tf.int32, (None), name='Y')
isTrain = tf.placeholder(tf.bool, shape=(), name='IsTraining')
learning_rate = tf.placeholder(tf.float32, shape=(), name = 'LearningRate')

one_hot_y = tf.one_hot(y, 43)

logits,_ = model(x, is_training=isTrain)
cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=one_hot_y, logits=logits)
loss_operation = tf.reduce_mean(cross_entropy)

top_predictions = tf.nn.top_k(tf.nn.softmax(logits), k=3)

optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate)
training_operation = optimizer.minimize(loss_operation)


# Danh gia mo hinh

correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(one_hot_y, 1))
accuracy_operation = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
saver = tf.train.Saver()

def evaluate(X_data, y_data, print_loss=False):
    num_examples = len(X_data)
    total_accuracy = 0
    total_loss=0
    sess = tf.get_default_session()
    image=None
    for offset in range(0, num_examples, BATCH_SIZE):
        batch_x, batch_y = X_data[offset:offset+BATCH_SIZE], y_data[offset:offset+BATCH_SIZE]
        accuracy,loss = sess.run([accuracy_operation,loss_operation], feed_dict={x: batch_x, y: batch_y, isTrain: False})
        total_accuracy += (accuracy * len(batch_x))
        total_loss+=(loss* len(batch_x))

    if print_loss:
        print('loss: ',total_loss/num_examples)

    return total_accuracy / num_examples


# Huan luyen mo hinh

# with tf.Session() as sess:
    
#     #get number of samples
#     num_examples = len(X_train)

#     #restore last saved model 
#     saver.restore(sess, tf.train.latest_checkpoint('.'))
    
#     #set the learning rate value to continue with training
#     rate=0.00001
#     count=0

#     print("Training... \n")
#     for i in range(EPOCHS):
#         X_process, Y_process = shuffle(X_train, y_train)
#         for offset in range(0, num_examples, BATCH_SIZE):
#             count+=1

#             end = offset + BATCH_SIZE
#             batch_x, batch_y = X_process[offset:end], Y_process[offset:end]
#             r=sess.run(training_operation, feed_dict={x: batch_x, 
#                                                       y: batch_y,
#                                                       isTrain: True, 
#                                                       learning_rate:rate})
            
#            # Print validation accuracy periodically on every 100 iterations 
#             if(count%100==0):
#                 validation_accuracy = evaluate(X_valid, y_valid, rate)
#                 print("---------------Validation Accuracy = {:.3f}".format(validation_accuracy))   
        
#         # Print train accuracy and validation accuracy after finishing training epoch 
#         train_accuracy = evaluate(X_process, Y_process)
#         validation_accuracy = evaluate(X_valid, y_valid, print_loss=True)
       
#         print("EPOCH {} ...".format(i+1))
#         print("Train Accuracy = {0}".format(train_accuracy))
#         print("Validation Accuracy = {0}".format(validation_accuracy))
#         print("learning_rate", rate)

#     test_accuracy = evaluate(X_test, y_test)
#     print("Test Accuracy = {0}".format(test_accuracy))

#     saver.save(sess, './lenet')
#     print("Model saved")


## Chay du doan.
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('.'))

    validation_accuracy = evaluate(X_valid, y_valid, print_loss=True)
    test_accuracy = evaluate(X_test, y_test, print_loss=True)
    train_accuracy = evaluate(X_train, y_train, print_loss=True)

    print('Train Accuracy', train_accuracy)
    print('Validation Accuracy', validation_accuracy)
    print('Test Accuracy', test_accuracy)


### Buoc 4: Kiem tra mo hinh va dua ra du doan.


# Load va xuat hinh anh.

def evaluate_custom(X_data, Y_data):

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint('.'))
        predictions, c_accuracy, c_loss = sess.run([top_predictions, accuracy_operation, loss_operation], feed_dict={x: X_data, y:Y_data,  isTrain: False})
        return predictions, c_accuracy, c_loss



from utils import *

def get_custom_dataset():
    
    directory = 'test_images'

    X_custom_set = []
    X_readable_set = []
    for i in range (1,5):
        image, readable_image = apply_model_to_image_raw_bytes(open(directory+'/'+str(i)+".jpg", "rb").read())
        X_custom_set.append(image)
        X_readable_set.append(readable_image)
        
    custom_set_X = np.array(X_custom_set)
    custom_set_Y = [1,17,30,14]
    return preprocess_dataset(custom_set_X), X_readable_set, custom_set_Y

# Load hinh anh
X_custom, X_custom_readable, Y_custom = get_custom_dataset()

# bieu do hinh anh
rows = 1
cols = 4
fig, axs = plt.subplots(rows,cols,squeeze=False)
fig.set_size_inches(25,15)
for i, image in enumerate(X_custom_readable):
    axs[int(i/cols)][i%cols].imshow(image)



## Du doan loai bien bao cho moi hinh anh

## Su dung mo hinh duoc huan luyen de dua ra du doan cho moi hinh anh.
predictions, c_accuracy, c_loss = evaluate_custom(X_custom, Y_custom)
rows = 1
cols = 4
fig, axs = plt.subplots(rows,cols,squeeze=False)
fig.set_size_inches(25,15)


# In ra cac ket qua du doan
for i, image in enumerate(X_custom_readable):
    axs[int(i/cols)][i%cols].imshow(image)

    title = (labels[predictions.indices[i][0]])
    axs[int(i/cols)][i%cols].set_title(title, fontsize=20)


## Phan tich hieu suat

print("Accuracy", c_accuracy*100)
print("Loss", c_loss)


# In ra 3 xac suat cao nhat cho cac du doan ve hinh anh bien bao giao thong.
rows = 1
cols = 4
fig, axs = plt.subplots(rows,cols,squeeze=False)
fig.set_size_inches(25,15)


# In cac du doan
for i, image in enumerate(X_custom_readable):
    axs[int(i/cols)][i%cols].imshow(image)

    values = (predictions.values[i]*100).tolist()
    indeces = (predictions.indices[i]).tolist()
    captions = list(map(lambda x:labels[x], indeces))
    
    title = '\n'.join('%s=%s%%' % t for t in zip(captions, values))
        
    axs[int(i/cols)][i%cols].set_title(title, fontsize=12)

plt.show()
