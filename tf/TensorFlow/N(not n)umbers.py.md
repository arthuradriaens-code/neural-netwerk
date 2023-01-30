## stop after 99% accuracy is reached
class myCallback(keras.callbacks.Callback): #code given to fit to stop when 99%

    def on_epoch_end(self,epoch,logs={}):

        if (logs.get('accuracy')>0.99):

            print("\nReached 99% accuracy so cancelling training!")

            self.model.stop_training = True

callbacks = myCallback()

model.fit(train_images,train_labels,epochs=10,callbacks=[callbacks])

### problem in naming
If you name it numbers.py it clashes with internal code.