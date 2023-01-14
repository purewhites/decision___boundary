import argparse
import importlib
from utils import *
import os.path as osp
import torch.nn as nn
from copy import deepcopy
from models.fact.helper import *
from dataloader.data_utils import *
import abc

'''
python baseline_decouple.py -projec fact -dataset cifar100  -base_mode "ft_cos" -new_mode "avg_cos" -gamma 0.1 -lr_base 0.1 -lr_new 0.1 -decay 0.0005 -epochs_base 600 -schedule Cosine -gpu 0 -temperature 16 -batch_size_base 256 -alpha 0.5 -start_session 1 -model_dir checkpoint/cifar100/fact/train/3/session0_max_acc.pth -job test

'''

MODEL_DIR= 'checkpoint/cub200/fact/ft_cos-avg_cos-data_init-start_0/2/session0_max_acc.pth'
DATA_DIR = '../../dataset/data/'
PROJECT = 'fact'

class Trainer(object, metaclass=abc.ABCMeta):
    def __init__(self, args):
        self.args = args
        self.args = set_up_datasets(self.args)
        self.dt, self.ft = Averager(), Averager()
        self.bt, self.ot = Averager(), Averager()
        self.timer = Timer()

        # train statistics
        self.trlog = {}
        self.trlog['train_loss'] = []
        self.trlog['val_loss'] = []
        self.trlog['test_loss'] = []
        self.trlog['train_acc'] = []
        self.trlog['val_acc'] = []
        self.trlog['test_acc'] = []
        self.trlog['max_acc_epoch'] = 0
        self.trlog['max_acc'] = [0.0] * args.sessions

    @abc.abstractmethod
    def train(self):
        pass

class FSCILTrainer(Trainer):
    def __init__(self, args):
        super().__init__(args)
        self.args = args
        self.set_save_path()
        self.args = set_up_datasets(self.args)

        self.model = MYNET(self.args, mode=self.args.base_mode)
        self.model = nn.DataParallel(self.model, list(range(self.args.num_gpu)))
        self.model = self.model.cuda()

        if self.args.model_dir is not None:
            print('Loading init parameters from: %s' % self.args.model_dir)
            self.best_model_dict = torch.load(self.args.model_dir)['params']
            
        else:
            print('random init params')
            if args.start_session > 0:
                print('WARING: Random init weights for new sessions!')
            self.best_model_dict = deepcopy(self.model.state_dict())

    def get_optimizer_base(self):
        optimizer = torch.optim.SGD(self.model.parameters(), self.args.lr_base, momentum=0.9, nesterov=True,
                                    weight_decay=self.args.decay)
        if self.args.schedule == 'Cosine':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.epochs_base)

        return optimizer, scheduler

    def get_dataloader(self, session):
        if session == 0:
            trainset, trainloader, testloader = get_base_dataloader(self.args)
        else:
            trainset, trainloader, testloader = get_new_dataloader(self.args, session)
        return trainset, trainloader, testloader

    def train(self):
        args = self.args
        t_start_time = time.time()

        result_list = [args]
        for session in range(args.start_session, args.sessions):
            train_set, trainloader, testloader = self.get_dataloader(session)
            self.model.load_state_dict(self.best_model_dict)
            
            if session == 0:  # load base class train img label
                print('new classes for this session:\n', np.unique(train_set.targets))
                optimizer, scheduler = self.get_optimizer_base()

                for epoch in range(args.epochs_base):
                    start_time = time.time()
                    # train base sess
                    tl, ta = base_train(self.model, trainloader, optimizer, scheduler, epoch, args)
                    tsl, tsa = test(self.model, testloader, epoch, args, session)

                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        self.trlog['max_acc_epoch'] = epoch
                        save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                        torch.save(dict(params=self.model.state_dict()), save_model_dir)
                        torch.save(optimizer.state_dict(), os.path.join(args.save_path, 'optimizer_best.pth'))
                        self.best_model_dict = deepcopy(self.model.state_dict())
                        print('********A better model is found!!**********')
                        print('Saving model to :%s' % save_model_dir)
                    print('best epoch {}, best test acc={:.3f}'.format(self.trlog['max_acc_epoch'],
                                                                       self.trlog['max_acc'][session]))

                    self.trlog['train_loss'].append(tl)
                    self.trlog['train_acc'].append(ta)
                    self.trlog['test_loss'].append(tsl)
                    self.trlog['test_acc'].append(tsa)
                    lrc = scheduler.get_last_lr()[0]
                    result_list.append(
                        'epoch:%03d,lr:%.4f,training_loss:%.5f,training_acc:%.5f,test_loss:%.5f,test_acc:%.5f' % (
                            epoch, lrc, tl, ta, tsl, tsa))
                    print('This epoch takes %d seconds' % (time.time() - start_time),
                          '\nstill need around %.2f mins to finish this session' % (
                                  (time.time() - start_time) * (args.epochs_base - epoch) / 60))
                    scheduler.step()

                result_list.append('Session {}, Test Best Epoch {},\nbest test Acc {:.4f}\n'.format(
                    session, self.trlog['max_acc_epoch'], self.trlog['max_acc'][session], ))

                if not args.not_data_init:
                    self.model.load_state_dict(self.best_model_dict)
                    self.model = replace_base_fc(train_set, testloader.dataset.transform, self.model, args)
                    best_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')
                    print('Replace the fc with average embedding, and save it to :%s' % best_model_dir)
                    self.best_model_dict = deepcopy(self.model.state_dict())
                    torch.save(dict(params=self.model.state_dict()), best_model_dir)

                    self.model.module.mode = 'avg_cos'
                    tsl, tsa = test(self.model, testloader, 0, args, session)
                    if (tsa * 100) >= self.trlog['max_acc'][session]:
                        self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                        print('The new best test acc of base session={:.3f}'.format(self.trlog['max_acc'][session]))

                self.dummy_classifiers = deepcopy(self.model.module.fc.weight.detach())
                
                self.dummy_classifiers = F.normalize(self.dummy_classifiers[self.args.base_class:,:],p=2,dim=-1)
                self.old_classifiers=self.dummy_classifiers[:self.args.base_class,:]

            else:  # incremental learning sessions
                print("training session: [%d]" % session)

                self.model.module.mode = self.args.new_mode
                self.model.eval()
                trainloader.dataset.transform = testloader.dataset.transform
                self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
                #self.model.module.update_adj(trainloader, np.unique(train_set.targets), session)
                #self.model.module.save_adj(args.save_path + f'/adj_session{session}.png')

                tsl, tsa = self.test_intergrate(self.model, testloader, 0, args, session, validation=True)

                self.trlog['max_acc'][session] = float('%.3f' % (tsa * 100))
                save_model_dir = os.path.join(args.save_path, 'session' + str(session) + '_max_acc.pth')

                self.best_model_dict = deepcopy(self.model.state_dict())
                print('Saving model to :%s' % save_model_dir)
                print('  test acc={:.3f}'.format(self.trlog['max_acc'][session]))

                result_list.append('Session {}, test Acc {:.3f}\n'.format(session, self.trlog['max_acc'][session]))

        result_list.append('Base Session Best Epoch {}\n'.format(self.trlog['max_acc_epoch']))
        result_list.append(self.trlog['max_acc'])
        print(self.trlog['max_acc'])
        save_list_to_txt(os.path.join(args.save_path, 'results.txt'), result_list)

        t_end_time = time.time()
        total_time = (t_end_time - t_start_time) / 60
        print('Base Session Best epoch:', self.trlog['max_acc_epoch'])
        print('Total time used %.2f mins' % total_time)

    def test_intergrate(self, model, testloader, epoch,args, session, validation=True):
        test_class = args.base_class + session * args.way
        model = model.eval()
        vl = Averager()
        va = Averager()
        lgt=torch.tensor([])
        lbs=torch.tensor([])

        with torch.no_grad():
            for i, batch in enumerate(testloader, 1):
                data, test_label = [_.cuda() for _ in batch]

                logits2 = model.module.forpass_fc(data)[:, :test_class]
                #logits2 = model.module.infer_a(data)[:, :test_class]
                logits=F.softmax(logits2,dim=1)
            
                loss = F.cross_entropy(logits, test_label)
                acc = count_acc(logits, test_label)
                vl.add(loss.item())
                va.add(acc)
                lgt=torch.cat([lgt,logits.cpu()])
                lbs=torch.cat([lbs,test_label.cpu()])
            vl = vl.item()
            va = va.item()
            print(f'epo {epoch}, test, loss={vl:.4f} acc={va:.4f}')
            
        return vl, va

    def l2_normolize(self, vector):
        l2 = np.sqrt(np.sum(vector * vector, axis=-1))
        return vector / l2

    def calculate_cos_std(self, testloader):
        tqdm_gen = tqdm(testloader)
        print(len(tqdm_gen))
        stds = []
        offsets = []
        l2_ds = []
        with torch.no_grad():
            for i, batch in enumerate(tqdm_gen):
                data, test_label = [_.cuda() for _ in batch]
                data = self.model.module.encode(data)
                data_l2 = data.cpu().numpy()
                data_l2 = np.sqrt(np.sum(data_l2 * data_l2, axis=1))
                l2_d = np.min(data_l2) / np.max(data_l2)
                l2_ds.append(l2_d)
                data = F.normalize(data, p=2, dim=-1)
                class_vectors = data.cpu().numpy()
                mean_vector = np.average(class_vectors, axis=0)
                mean_vector = self.l2_normolize(mean_vector)
                support_vector = F.normalize(self.model.module.fc.weight.data[i, :], p=2, dim=-1).cpu().numpy()
                offset = np.sum(mean_vector * support_vector, axis=0)
                offsets.append(offset)
                cos_dis = np.dot(class_vectors, mean_vector[:, np.newaxis])
                cos_std = np.average(cos_dis, axis=0)
                stds.append(cos_std[0])

        return stds, offsets, l2_ds

    def test_report(self):
        args = self.args
        self.best_model_dict = {k:v for k,v in self.best_model_dict.items() if 'encoder' in k}
        self.model.load_state_dict(self.best_model_dict, strict = False)
        ensure_path(args.save_path + '/CM')
        f = open(f'{args.save_path}/test_report.txt', 'w')
        txt_writer = [str(self.args)+'\n']
        acc_list = []
        acc2_list = []
        labels = [a[:-1] for a in open(f'data/{args.dataset}_labels.txt').readlines()]

        # #self.model.module.load_state_dict()
        # save_state_dict = self.best_model_dict
        # model_dict = self.model.module.state_dict()
        # new_dict = {k:v for k,v in save_state_dict.items() if 'encoder' in k}
        # #model_dict.update(new_dict)
        # self.model.module.load_state_dict(model_dict)
        self.model.module.mode = self.args.new_mode
        self.model.eval()
        # trainset, trainloader, testloader = get_dataloader(args, 0)
        # replace_base_fc(trainset, testloader.dataset.transform, self.model, args)

        # for cls in np.arange(60):
        #     train_set, trainloader, testloader = get_cls_dataloader(self.args, cls)
        #     trainloader.dataset.transform = testloader.dataset.transform
        #     self.model.module.update_fc_cls(trainloader, cls)

        for session in range(args.start_session, args.sessions):

            train_set, trainloader, testloader = self.get_dataloader(session)
            trainloader.dataset.transform = testloader.dataset.transform
            # self.model.module.update_fc(trainloader, np.unique(train_set.targets), session)
            test_class = args.base_class + session * args.way

            CM = np.zeros((test_class, test_class))
            acc_l = 0
            acc2_l = 0
            with torch.no_grad():
                tqdm_gen = tqdm(testloader)
                for i, batch in enumerate(tqdm_gen, 1):
                    data, test_label = [_.cuda() for _ in batch]

                    logits = self.model.module.forpass_fc(data)
                    logits = logits[:, :test_class]
                    loss = F.cross_entropy(logits, test_label)
                    acc = count_acc(logits, test_label)
                    acc_l += acc
                    pred = torch.argmax(logits, dim=-1)
                    for gt, pr in zip(test_label, pred):
                        CM[gt][pr] += 1

            acc_list.append(acc_l / len(tqdm_gen))
            #acc2_list.append(acc2_l / len(tqdm_gen))
            plt.clf()
            test_s = np.sum(CM, axis=1)
            test_s = test_s[:, np.newaxis]
            CM = CM / test_s
            plt.imshow(CM, cmap=plt.cm.Blues)
            plt.savefig(f'{args.save_path}/CM/session{session}.png', dpi=300)
            np.savetxt(f'{args.save_path}/CM.csv', CM, delimiter=',')

            print(acc_list)

            if session == args.sessions - 1:

                TP = [CM[i][i] for i in range(args.num_classes)]
                TPn = np.array(TP)  # / test_size
                txt_writer.append('\nAccuracy\n' + str(acc_list) + '\n')
                txt_writer.append(f'old average: {(sum(TPn[:args.base_class]) / args.base_class) * 100:.2f}% \n')
                txt_writer.append(
                    f'new average: {sum(TPn[args.base_class:]) / (args.num_classes - args.base_class) * 100 :.2f}%\n\n')
                txt_writer.append(
                    f'基础类别间平均混淆: {(CM[:args.base_class, :args.base_class].sum() - sum(TP[:args.base_class])) / (args.base_class - 1) * 100:.2f}%\n')
                txt_writer.append(
                    f'new to old: {(CM[args.base_class:, :args.base_class].sum() / (args.num_classes - args.base_class) * 100):.2f}%\n')
                txt_writer.append(
                    f'old to new: {(CM[:args.base_class, args.base_class:].sum() / args.base_class * 100):.2f}%\n')
                txt_writer.append(
                    f'new to new: {((CM[args.base_class:, args.base_class:].sum() - sum(TP[args.base_class:])) / (args.num_classes - args.base_class) * 100):.2f}%\n')

                # ---------- average precision ---------- #
                st = np.argsort(np.argsort(TPn))
                plt.clf()
                plt.bar(st[:args.base_class], TPn[:args.base_class], color='#FFE4C4')
                plt.bar(st[args.base_class:], TPn[args.base_class:], color='#E4C4FF')
                plt.savefig(f'{args.save_path}/TP_classes.png', dpi=300)
                stds, offsets, l2_ds = self.calculate_cos_std(testloader)
                stdstmp = [f'{std:.3f}' for std in stds]
                offsetstmp = [f'{offset:.3f}' for offset in offsets]
                l2_dstmp = [f'{l2_d:.3f}' for l2_d in l2_ds]
                txt_writer.append('\nstds\n' + str(stdstmp) + '\n\noffsets\n' + str(offsetstmp) + '\n\nl2_ds\n' + str(
                    l2_dstmp) + '\n\n')
                plt.clf()
                plt.plot(list(range(args.num_classes)), stds, color='#00FF00', label='std')
                plt.plot(list(range(args.num_classes)), offsets, color='#FF0000', label='offset')
                plt.plot(list(range(args.num_classes)), l2_ds, color='#0000FF', label='l2_d')
                plt.savefig(f'{args.save_path}/std_analysis.png', dpi=300)
                txt_writer.append(
                    f'old average cosine-std on val: {sum(stds[:args.base_class]) / args.base_class:.4f}\n')
                txt_writer.append(
                    f'new average cosine-std on val: {sum(stds[args.base_class:]) / (args.num_classes - args.base_class):.4f}\n')
                txt_writer.append(
                    f'old average cosine-offset on val: {sum(offsets[:args.base_class]) / args.base_class:.4f}\n')
                txt_writer.append(
                    f'new average cosine-offset on val: {sum(offsets[args.base_class:]) / (args.num_classes - args.base_class):.4f}\n')

                # ---------- top low precision ---------- #
                txt_writer.append('\n Top low precision \n')
                for i, ind in enumerate(np.argsort(TPn)):
                    flag = 'old' if ind < args.base_class else 'new'
                    txt_writer.append(f'{i + 1:02d} {ind:02d}.{labels[ind]} {TPn[ind]} {flag} \n')

                # --------- top confusion -------------#
                topnum = 20
                txt_writer.append(f'\nTop {topnum} confusion between class:\n')
                CM_tmp = CM
                for i in range(args.num_classes):
                    CM_tmp[i][i] = 0

                confusion_list = CM_tmp.reshape((-1,))
                for i, index in enumerate(list(np.argsort(-confusion_list)[:topnum])):
                    pred, gt = index % args.num_classes, index // args.num_classes
                    conf = confusion_list[index]
                    txt_writer.append(f'{i}. confusion:{conf} of {gt}.{labels[gt]} to {pred}.{labels[pred]}\n')

                # ---------support vector cosine matrix-------------#

                fc_data = self.model.module.fc.weight.data
                support_vectors = F.normalize(fc_data, p=2, dim=-1).cpu().numpy()
                np.savetxt(f'{args.save_path}/support_vector.csv', support_vectors, delimiter=',')
                cos_mat = np.dot(support_vectors, np.transpose(support_vectors, (1, 0)))
                maxx = np.max(cos_mat.reshape((-1,)))
                minn = np.min(cos_mat.reshape((-1,)))
                for i in range(args.num_classes):
                    cos_mat[i][i] = minn
                plt.clf()
                plt.imshow((cos_mat - minn) / (maxx - minn), cmap=plt.cm.Greens)
                plt.savefig(f'{args.save_path}/cos_mat.png', dpi=300)
                np.savetxt(f'{args.save_path}/cos_mat.csv', cos_mat, delimiter=',')
                txt_writer.append(f'\nTop {topnum} cosine distance between class:\n')

                cos_mat_tmp = cos_mat.copy()
                cos_mat_tmp = (cos_mat_tmp - minn) / (maxx - minn)
                for i in range(args.num_classes - 1):
                    cos_mat[i, i + 1:] = 0
                cosine_list = cos_mat.reshape((-1,))
                for i, index in enumerate(list(np.argsort(-cosine_list)[:topnum])):
                    pred, gt = index % args.num_classes, index // args.num_classes
                    cosine = cosine_list[index]
                    txt_writer.append(f'{i}. cosine:{cosine:.4f} between {gt}.{labels[gt]} and {pred}.{labels[pred]}\n')

                # calculate diversion and analysis result relation
                normolize = lambda x: (x - np.min(x)) / (np.max(x) - np.min(x))
                cos_div = 1 / np.sum(cos_mat_tmp, axis=1)
                cos_div = normolize(cos_div)
                conf_div = np.sum(CM, axis=0)
                conf_div = normolize(conf_div)
                plt.clf()
                plt.plot(list(range(args.num_classes)), TPn, color='#FF9078', label='Class accuracy')
                plt.plot(list(range(args.num_classes)), cos_div, color='#90FF78', label='Cosine confusion')
                plt.plot(list(range(args.num_classes)), conf_div, color='#9078FF', label='Predict confusion')
                plt.savefig(f'{args.save_path}/analysis.png', dpi=300)

                plt.clf()
                plt.bar(st, cos_div, color='#90FF78', alpha=0.6)
                plt.bar(st, conf_div, color='#9078FF', alpha=0.4)
                plt.savefig(f'{args.save_path}/analysis2.png', dpi=300)

                plt.clf()
                plt.bar(st, stds, color='#00FF00', label='cos-std', alpha=0.6)
                plt.bar(st, offsets, color='#FF0000', label='cos-offset', alpha=0.4)
                plt.savefig(f'{args.save_path}/analysis3.png', dpi=300)

                f.writelines(txt_writer)
                f.close()


    def set_save_path(self):

        self.args.save_path = 'checkpoint/%s/' % self.args.dataset + '%s/' % self.args.project + '%s/' % self.args.job
        ensure_path(self.args.save_path)
        ll = os.listdir(self.args.save_path)
        if ll == []:
            new_id = 1
        else:
            new_id = max(list(map(int, ll))) + 1
        self.args.save_path = self.args.save_path + str(new_id)
        ensure_path(self.args.save_path)

        return None

    

def get_command_line_parser():
    parser = argparse.ArgumentParser()
    # about dataset and network
    parser.add_argument('-project', type=str, default='fact')
    parser.add_argument('-dataset', type=str, default='cub200',
                        choices=['mini_imagenet', 'cub200', 'cifar100'])
    parser.add_argument('-dataroot', type=str, default=DATA_DIR)
    parser.add_argument('-epochs_base', type=int, default=800)
    parser.add_argument('-epochs_new', type=int, default=100)
    parser.add_argument('-lr_base', type=float, default=0.1)
    parser.add_argument('-lr_new', type=float, default=0.1)
    parser.add_argument('-schedule', type=str, default='Cosine', choices=['Step', 'Milestone','Cosine'])
    parser.add_argument('-decay', type=float, default=0.0005)
    parser.add_argument('-momentum', type=float, default=0.9)
    parser.add_argument('-gamma', type=float, default=0.1)
    parser.add_argument('-temperature', type=float, default=16)
    parser.add_argument('-not_data_init', action='store_true', help='using average data embedding to init or not')
    parser.add_argument('-batch_size_base', type=int, default=256)
    parser.add_argument('-batch_size_new', type=int, default=0, help='set 0 will use all the availiable training image for new')
    parser.add_argument('-test_batch_size', type=int, default=10)
    parser.add_argument('-base_mode', type=str, default='ft_cos', choices=['ft_dot', 'ft_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier
    parser.add_argument('-new_mode', type=str, default='avg_cos', choices=['ft_dot', 'ft_cos', 'avg_cos']) # ft_dot means using linear classifier, ft_cos means using cosine classifier, avg_cos means using average data embedding and cosine classifier
    #for fact
    parser.add_argument('-alpha', type=float, default=0.0)
    parser.add_argument('-start_session', type=int, default=1)
    parser.add_argument('-model_dir', type=str, default=MODEL_DIR, help='loading model parameter from a specific dir')
    parser.add_argument('-set_no_val', action='store_true', help='set validation using test set or no validation')
    # about training
    parser.add_argument('-num_gpu', default=1)
    parser.add_argument('-job', default='test')
    parser.add_argument('-gpu', default='1')
    parser.add_argument('-num_workers', type=int, default=8)
    parser.add_argument('-seed', type=int, default=1)
    parser.add_argument('-debug', action='store_true')
    return parser

parser = get_command_line_parser()
args = parser.parse_args()
set_seed(args.seed)
pprint(vars(args))

trainer = FSCILTrainer(args)
trainer.test_report()