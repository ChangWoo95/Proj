import string
import argparse

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data
import torch.nn.functional as F

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def demo(opt):
    """ model configuration """
    lists = [] #목적지라고 생각하는 사진에서 인식한 text를 담을 배열

    converter = AttnLabelConverter(opt.character) #ATTN

    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3

    model = Model(opt) #model.py의 Model import

    print('model input parameters', opt.imgH, opt.imgW, opt.num_fiducial, opt.input_channel, opt.output_channel,
          opt.hidden_size, opt.num_class, opt.batch_max_length, opt.Transformation, opt.FeatureExtraction,
          opt.SequenceModeling, opt.Prediction) #파라미터 값 정보 출력

    model = torch.nn.DataParallel(model).to(device) #GPU로 데이터 병렬 처리 진행 

    # load model
    print('loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=device)) #모델의 매개변수를 불러옴

    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data1 = RawDataset(root=opt.image_folder1, opt=opt)  # use RawDataset 간판탐지결과
    demo_data2 = RawDataset(root=opt.image_folder2, opt=opt)  # use RawDataset 구글맵문자열탐지결과

    demo_loader1 = torch.utils.data.DataLoader(
        demo_data1, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    demo_loader2 = torch.utils.data.DataLoader(
        demo_data2, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)

    # predict
    model.eval()
    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader1:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)

            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            #ATTn
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a') #이어서 쓸수 있게 열고
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}') #테이블 양식 출력
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n') #txt에 테이블 양식 저장

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # calculate confidence score (= multiply of pred_max_prob) confidence score 값을 계산
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]

                lists.append(pred)
                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}') #구한 값을 출력
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n') #구한 값을 txt에 저장
            
            log.close() #파일 닫기

    with torch.no_grad():
        for image_tensors, image_path_list in demo_loader2:
            batch_size = image_tensors.size(0)
            image = image_tensors.to(device)
            # For max length prediction
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size).to(device)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0).to(device)

            #ATTn
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)


            log = open(f'./log_demo_result.txt', 'a') #이어서 쓸수 있게 열고
            dashed_line = '-' * 80
            head = f'{"image_path":25s}\t{"predicted_labels":25s}\tconfidence score'
            
            print(f'{dashed_line}\n{head}\n{dashed_line}') #테이블 양식 출력
            log.write(f'{dashed_line}\n{head}\n{dashed_line}\n') #txt에 테이블 양식 저장

            preds_prob = F.softmax(preds, dim=2)
            preds_max_prob, _ = preds_prob.max(dim=2)
            for img_name, pred, pred_max_prob in zip(image_path_list, preds_str, preds_max_prob):
                pred_EOS = pred.find('[s]')
                pred = pred[:pred_EOS]  # prune after "end of sentence" token ([s])
                pred_max_prob = pred_max_prob[:pred_EOS]

                # confidence score 값을 계산
                confidence_score = pred_max_prob.cumprod(dim=0)[-1]


                print(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}') #구한 값을 출력
                log.write(f'{img_name:25s}\t{pred:25s}\t{confidence_score:0.4f}\n') #구한 값을 txt에 저장
                if pred in lists :
                    print(pred + "은(는) 알맞은 목적지입니다.")
                else : 
                    print(pred + "은(는) 알맞은 목적지가 아닙니다.")            

            log.close() #파일 닫기