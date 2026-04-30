import datetime, logging
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Times New Roman"
import pandas as pd
import os

def visualize(train_acc_list,train_loss_list,test_acc_list,test_loss_list,training_time_list,inference_time_list,training_experiment_name,training_time,dataset,experiment_style,is_async,epoch,sensitive,ixx_r,ixy_r,ixz_r):
	# Export data
	current_dir = os.getcwd()

	dataset_dir = os.path.join(current_dir, dataset)

	if not os.path.isdir(dataset_dir):
		os.mkdir(dataset_dir)

	experiment_style_dir =os.path.join(dataset_dir, experiment_style)

	if not os.path.isdir(experiment_style_dir):
		os.mkdir(experiment_style_dir)

	fig_dir = os.path.join(experiment_style_dir, "exp_figure")
	log_dir = os.path.join(experiment_style_dir, "exp_log")
	#model_dir = os.path.join(experiment_style_dir, "exp_model")

	if not os.path.isdir(fig_dir):
		os.mkdir(fig_dir)
	if not os.path.isdir(log_dir):
		os.mkdir(log_dir)
	'''
	if not os.path.isdir(model_dir):
		os.mkdir(model_dir)
	'''
	training_experiment_date = datetime.datetime.now()
	training_experiment_date = training_experiment_date.strftime("%m/%d/%Y, %H:%M:%S")
	training_experiment_date = training_experiment_date.replace(" ", "_")
	training_experiment_date = training_experiment_date.replace(":", "_")
	training_experiment_date = training_experiment_date.replace("/", "_")
	training_experiment_date = training_experiment_date.replace(",", "")

	if is_async:
		excel_path = log_dir+'/async_{}_{}.xlsx'.format(training_experiment_name, training_experiment_date)
		csv_path = log_dir+'/async_{}_{}.csv'.format(training_experiment_name, training_experiment_date)
		writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')
	else:
		excel_path = log_dir + '/sync_{}_{}.xlsx'.format(training_experiment_name, training_experiment_date)
		csv_path = log_dir + '/sync_{}_{}.csv'.format(training_experiment_name, training_experiment_date)
		writer = pd.ExcelWriter(excel_path, engine='xlsxwriter')

	df2 = pd.DataFrame()
	df2['Training Loss'] = train_loss_list
	df2['Test Loss'] = test_loss_list
	df2['Training Accuracy'] = train_acc_list
	df2['Test Accuracy'] = test_acc_list
	df2['Training time list'] = training_time_list
	df2['Inference time list'] = inference_time_list
	df2['Training Time'] = training_time
	df2['Total Epoch'] = epoch
	df2['Sensitive attribute'] = sensitive
	df2['Coefficient x'] = ixx_r
	df2['Coefficient r'] = ixy_r
	df2['Coefficient s'] = ixz_r
	df2.to_excel(writer, sheet_name='loss_acc', index=True)
	df2.to_csv(csv_path, index=False, header=True)

	writer.save()
	logging.info("Saved Excel to: {} \n".format(excel_path))
	logging.info("Saved CSV to: {} \n".format(csv_path))

	'''
	model_file = model_dir+'/{}_{}users_{}dynamic_both_{}dynamic_batch_{}.log'.format(training_experiment_name, users, dynamic_both,dynamic_batch,training_experiment_date)
	with open(model_file, "w") as f:
		f.write(client_model)

	'''
	# Draw Figure

	plt.rcParams["font.family"] = "Times New Roman"

	# 3. Draw figures ==================================================================
	markers = {'Train Loss': '^', 'Train Acc.': 'o', 'Test Acc.': 's'}
	x = np.arange(len(train_loss_list))
	# print("x.length:", len(x))
	fig = plt.figure(figsize=(7, 8))
	ax1 = fig.add_subplot(211)

	ax1.set_xlabel('Training Progress')
	ax1.set_ylabel('Accuracy')
	print(train_acc_list)
	print(test_acc_list)
	ax1.plot(x, train_acc_list, label='Training Acc')
	ax1.plot(x, test_acc_list, label='Test Acc')
	# ax1.tick_params(axis='y')
	# plt.ylim(0, 1.0)
	plt.legend(loc='best')
	plt.title("Train Loss: {:.4f}, Train Acc: {}, Test Loss: {:.4f}, Test Acc: {}".format(train_loss_list[-1],train_acc_list[-1],test_loss_list[-1],test_acc_list[-1]))
	plt.grid(linestyle='--')
	ax2 = fig.add_subplot(212)
	ax2.set_ylabel('Loss')  # we already handled the x-label with ax1
	ax2.plot(x, train_loss_list, label='Training Loss')
	ax2.plot(x, test_loss_list, label='Test Loss')
	# ax2.tick_params(axis='y')
	# plt.ylim(0, 5.0)
	plt.legend(loc='best')

	# fig.tight_layout()  # otherwise the right y-label is slightly clipped
	# plt.grid(color='r', linestyle='-', linewidth=2)
	plt.grid(linestyle='--')
	# plt.title("Loss: {:.4f}, Train Acc: {}, Test Acc: {}".format(trainer.train_loss_list[-1], trainer.train_acc_list[-1], trainer.test_acc_list[-1]))

	if is_async:
		figure_name = fig_dir+"/async_{}_{}.eps".format(training_experiment_name,training_experiment_date)  # CIFAR-10
	else:
		figure_name = fig_dir + "/sync_{}_{}.eps".format(training_experiment_name, training_experiment_date)
	logging.info("Saved Figure to: {} \n".format(figure_name))

	#logging.disable(logging.INFO)

	plt.savefig(figure_name, format='eps', dpi=1000)
	plt.show()