from utility.parser import init_args, save_parms, format_args
from utility.functions import set_seed

args = init_args()


from train import Trainer


set_seed(args.seed)

if args.if_train_teacher:
    trainer = Trainer(args)
    trainer.train_teacher()

elif args.is_train_student:
    trainer = Trainer(args)
    trainer.train_teacher_assistant()
elif not (args.if_train_teacher or args.is_train_student):
    trainer = Trainer(args)
    trainer.train_student()
else:
    raise NotImplementedError()
