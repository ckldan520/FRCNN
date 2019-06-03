import keras
from keras import layers
from importlib import import_module
import os
from loss_pack import losses
from keras.optimizers import Adam


class Model():
    def __init__(self, args):
        self.args = args
        self.rpn_model, self.classifier_model, self.model_all = self.build_model(self.args)


    def build_model(self, args):
        module = import_module('model.' + args.model.lower())


        num_anchors = len(args.anchor_box_scales) * len(args.anchor_box_ratios)
        img_input = keras.Input(shape=(None, None, args.n_dims))
        roi_input = keras.Input(shape=(None, 4))
        #feature extract
        base_layer = module.nn_base(args, img_input)
        #rpn
        rpn = module.rpn(base_layer, num_anchors)
        rpn_model = keras.Model(inputs=img_input, outputs=rpn[:2])
        #classifier
        classifier = module.classifier(base_layer, roi_input, args.num_rois, nb_classes = args.class_len)
        classifier_model = keras.Model([img_input, roi_input], outputs=classifier)
        #total model
        model_all = keras.Model([img_input, roi_input], rpn[:2]+classifier)


        #load model
        try:
            print('loading weights from {}'.format(self.args.pretrained_model))
            rpn_model.load_weights(self.args.pretrained_model, by_name=True)
            classifier_model.load_weights(self.args.pretrained_model, by_name=True)
        except:
            print('Could not load pretrained model weights')


        #compile
        rpn_model.compile(optimizer=Adam(lr=args.lr), loss=[losses.rpn_loss_cls(num_anchors), losses.rpn_loss_regr(num_anchors)])
        classifier_model.compile(optimizer=Adam(args.lr), loss=[losses.class_loss_cls, losses.class_loss_regr(args.class_len - 1)],
                                 metrics={'dense_class_{}'.format(args.class_len): 'accuracy'})
        model_all.compile(optimizer='sgd', loss='mae')



#        keras.utils.plot_model(rpn_model, to_file=os.path.join('./', "rpn_model.png"),
#                                      show_shapes=True)
#        keras.utils.plot_model(classifier_model, to_file=os.path.join('./', "classifier_model.png"),
#                               show_shapes=True)
        return rpn_model, classifier_model, model_all
