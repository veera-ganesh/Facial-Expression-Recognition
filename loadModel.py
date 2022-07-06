# import os
#
# import tensorflow
# from tensorflow.keras.models import Sequential, load_model
# cd = os.getcwd()
# print(cd)
# #model = Sequential()
# #model = tensorflow.keras.models.load_model(cd+'\\modelSaved\\saved_model.pb')
# new_model = tensorflow.keras.models.load_model(cd+'\\modelSaved')
# print(os.listdir(cd+'\\modelSaved'))
# new_model.summary()
# # loss, acc = model.evaluate(test_images, test_labels, verbose=2)
# # print('Restored model, accuracy: {:5.2f}%'.format(100 * acc))

import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(4), [1,2,3,4], 'r', label='Perplexity')
plt.plot(range(4), [5,6,7,8], 'b', label='Validation Perplexity')
plt.title('Perplexity and Validation Perplexity')
plt.legend()
plt.show()