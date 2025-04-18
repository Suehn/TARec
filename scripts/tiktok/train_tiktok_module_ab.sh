python main.py --dataset tiktok --batch_size 1024 --lr 1e-3 --early_stopping_patience 8 \
  --teacher_model_type "gcl" --is_softmax "True" --embed_size 64 --n_layers 2 \
  --if_train_teacher True \
  

python main.py --dataset tiktok --batch_size 4096 --student_lr 2e-3 --early_stopping_patience 8 \
  --teacher_model_type "gcl" --is_softmax "True" --teacher_assistant_model_type "lightgcl" \
  --embed_size 64 --student_embed_size 64 --n_layers 2 --ta_n_layers 1 \
  --teacher_model_dict_name "teacher_model_great" \
  --cl_loss_rate 1e-4  --kd_loss_rate 1e-6 \
  --is_train_student True \
  

python main.py  --name "10-cl-ta-skkd" --dataset tiktok --batch_size 4096 --student_lr 1.5e-3  --early_stopping_patience 48 \
  --teacher_model_type "gcl" --is_softmax "True" \
  --teacher_assistant_model_type "lightgcl" --student_model_type "mlpgcl" \
  --embed_size 64 --student_embed_size 64 --n_layers 2 --student_n_layers 1 \
  --teacher_model_dict_name "teacher_model_great" \
  --teacher_assistant_model_dict_name "ta_model_great" \
  --cl_loss_rate 1e-3  \
  --kd_loss_rate 1e-2 \

