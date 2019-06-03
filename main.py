from option import args
import data_loaders
import model
import trainer


loader = data_loaders.DataLoader(args)
args.class_len = loader.args.class_len
network = model.Model(args)
t = trainer.Trainer(args, loader, network)
while not t.terminate():
    t.train()

print('Finished!!!')