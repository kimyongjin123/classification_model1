import cv2
import numpy as np

from 이미지분류프로젝트 import readImageDirect
def readImageDirect(rpath):
    cnt = 0
    f_lists = os.listdir(rpath)
    for folder in f_lists:
        f_names = os.listdir(rpath + "\\" + folder)
        print(folder, ":", end="")
        for f_name in f_names:
            ori_img = cv.imread(rpath + "\\" + folder + "\\" + f_name)
def removeBackgroundFolder(rpath):
            # cv.destroyAllWindows()
            print(".",end="")
            print()
def singleRemoveBackGround(imagePathName):
    for f_name in f_names:
        ori_img = cv.imread(rpath + "\\" +folder +"\\" +f_name)
        ori_img = cv.resize(ori_img,(256,256))
        for ix in range(5):
            arg_img = imageAugment_sub(ori_img)
            cv.imwrite(rpath + "\\" + folder + "\\" + str(cnt)+f_name )
            cnt+=1
        print(".",end="")
    print()
def load_directory(rootpath):#{label:[이미지 리스트]}
    f_lists = os.listdir(rootpath)
    print(f_lists)
    y_labels = []
    x_files = []
    for label,fpath in enumerate(f_lists):
        print(".", end="")
        f_name = r"{}\{}".format( rootpath,label)
        f_names = os.listdir(f_name)
        #print(f_names)
        for p in f_names:
            y_labels.append(label)
            fimg = cv.imread(r"{}\{}".format(f_name,p))
            fimg = cv.cvtColor(fimg,cv2.COLOR_BGR2RGB)
            fimg = cv.resize(fimg,(64,64))
            x_files.append(fimg)
    return f_lists,np.array(y_labels),np.array(x_files)
def getTrainData(dpath):
    label_list, y_data, x_data = load_directory(dpath)
    print(y_data.shape)
    print(x_data.shape)
    print(len(label_list))
    print(label_list[0])
    # shuffle
    from sklearn.model_selection import train_test_split  # pycharm version 3.11
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=0.2, random_state=10, stratify=y_data)
    return {"label_list":label_list,"x_train":(x_train,y_train),
            "test":(x_test,y_test)}

if __name__=="__main__":
     readImageDirect(r"D:\ imgs")#  데이터 증강 호출
