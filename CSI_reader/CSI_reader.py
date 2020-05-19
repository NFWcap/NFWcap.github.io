# -*- coding: utf-8 -*-


from struct import unpack,pack
import numpy as np
from math import sqrt
import math
from itertools import cycle
import matplotlib.pyplot as plt
from scipy import optimize

subcarriers_index = [-28,-26,-24,-22,-20,-18,-16,-14,-12,-10,-8,-6,-4,-2,-1,1,3,5,7,9,11,13,15,17,19,21,23,25,27,28]

class CSI(object):

    def __init__(self,filename,values):
        self.filename = filename
        self.value = values
        self.csi_data = self.read_csi_data()
        self.raw_phase = self.phase_caculate()
        self.amplitude = self.amplitude_caculate()
        self.unwraped_phase = self.phase_unwrap() #相位角
        self.atf_lpf = self.lpf_filter()
        self.finger_print = self.get_fingerprinting_list()
        self.amplitude_print = self.get_amplitude_list()


    @staticmethod
    def expandable_or(a,b):
        r = a | b
        low = r & 0xff
        return unpack('b',pack('B',low))[0]

    @staticmethod
    def dbinv(x):
        return 10**(x/10)

    def total_rss(self,data):
        rssi_mag = 0
        if data['rssi_a'] != 0:
            rssi_mag = rssi_mag + (data['rssi_a'])
        if data['rssi_b'] != 0:
            rssi_mag = rssi_mag + self.dbinv(data['rssi_b'])
        if data['rssi_c'] != 0:
            rssi_mag = rssi_mag + self.dbinv(data['rssi_c'])
        return 10*np.log10(rssi_mag)-44-data['agc']

    def read_csi_data(self):
        file_name = self.filename
        data_stac = []
        triangle = np.array([1,3,6])   #三角？
        #读取dat文件内容
        with open(file_name,'rb') as f:
            buff = f.read()
            count = 0
            curr = 0
            while curr < (len(buff) - 3):
                data_len = unpack('>h',buff[curr:curr + 2])[0]
                code = unpack('B',buff[curr + 2:curr + 3])[0]
                curr = curr + 3
                if code == 187:
                    data = self.read_bfree(buff[curr:])
                    perm = data['perm']
                    nrx = data['nrx']
                    csi = data['csi']
                    if sum(perm) == triangle[nrx-1]:
                        csi[:,perm-1,:] = csi[:,[x for x in range(nrx)],:]
                    csi_matrix = self.scaled_csi(data)
                    data['csi'] = csi_matrix
                    data_stac.append(data)
                    count = count +1
                curr = curr + data_len -1

        return data_stac

    def read_bfree(self,array):

        result = {}
        array = array
        timestamp_low = array[0] + (array[1] << 8) + (array[2] << 16) + (array[3] << 24)
        bf_count = array[4] + (array[5] << 8)
        nrx = array[8]
        ntx = array[9]
        rssi_a = array[10]
        rssi_b = array[11]
        rssi_c = array[12]
        noise = unpack('b',pack('B',array[13]))[0]
        agc = array[14]
        antenna_sel = array[15]
        len_sum = array[16] +(array[17] << 8)
        fake_rate_n_flags = array[18] + (array[19] << 8)
        calc_len = (30* (nrx * ntx * 8 * 2 + 3)+ 7) // 8
        payload = array[20:]
        if len_sum != calc_len:
            print("数据发现错误！")
            exit(0)

        result['timestamp_low'] = timestamp_low
        result['bf_count'] = bf_count
        result['rssi_a'] = rssi_a
        result['rssi_b'] = rssi_b
        result['rssi_c'] = rssi_c
        result['nrx'] = nrx
        result['ntx'] = ntx
        result['agc'] = agc
        result['fake_rate_n_flags'] = fake_rate_n_flags
        result['noise'] = noise

        csi = np.zeros((ntx,nrx,30),dtype = np.complex64)

        idx = 0
        for sub_idx in range(30):
            idx = idx +3
            remainder = idx % 8
            for r in range(nrx):
                for t in range(ntx):
                    real = self.expandable_or((payload[idx // 8] >> remainder),
                                              (payload[idx // 8 + 1] << (8 - remainder)))
                    img = self.expandable_or((payload[idx // 8 + 1] >> remainder),
                                              (payload[idx // 8 + 2] << (8 - remainder)))
                    csi[t,r,sub_idx] = complex(real,img)
                    idx = idx + 16

        result['csi'] = csi
        perm = np.zeros(3,dtype = np.uint32)
        perm[0] = (antenna_sel & 0x3) + 1
        perm[1] = ((antenna_sel >> 2) & 0x3) +1
        perm[2] = ((antenna_sel >> 4) & 0x3) +1
        result['perm'] = perm
        return result

    def scaled_csi(self,data):

        csi = data['csi']
        ntx = data['ntx']
        nrx = data['nrx']

        csi_sq = csi * np.conj(csi)
        csi_pwr = csi_sq.sum().real
        rssi_pwr = self.dbinv(self.total_rss(data))
        scale = rssi_pwr / (csi_pwr / 30)

        if data['noise'] == -127:
            noise = -92
        else:
            noise = data['noise']
        thermal_noise_pwr = self.dbinv(noise)

        quant_error_pwr = scale * (nrx*ntx)
        total_nois_pwr = thermal_noise_pwr + quant_error_pwr
        ret = csi *sqrt(scale / total_nois_pwr)

        if ntx == 2:
            ret = ret * sqrt(2)
        elif ntx == 3:
            ret = ret * sqrt(self.dbinv(4.5))

        return ret

    def phase_caculate(self,x=1,y=1):
        """
        输入：帧信息数据集合
        输出：相位集合
        """
        frame_phase_list = []
        for i in self.csi_data:
            csi_matrix = i['csi'][x-1][y-1]
            raw_phase = np.angle(csi_matrix)
            frame_phase_list.append(raw_phase)
        return frame_phase_list

    def amplitude_caculate(self,x=1,y=1):
        """
        输入：帧信息数据集合
        输出：幅度集合
        """
        frame_amplitude_list = []
        for i in self.csi_data:
            csi_matrix = i['csi'][x-1][y-1]
            amplitude = np.abs(csi_matrix)
            frame_amplitude_list.append(amplitude)
        return frame_amplitude_list

    def phase_unwrap(self):
        """
        输入：未解缠绕的帧的相位集合
        输出：解缠绕后的帧的相位结合
        """
        unwraped_phase_list = list()
        for i in self.raw_phase:
            unwraped_phase_list.append(np.unwrap(i))
        return unwraped_phase_list

    def lpf_filter(self):
        """
        输入：各个帧的相位集合组成的集合，以及子载波序列[-28,-26,……]
        输出：符合条件的帧的相位集合组成的集合
        """
        fg_frame = []
        frame_len = len(self.unwraped_phase)
        for i in range(frame_len):
            r_list = list()
            for j in range(29):
                r = (self.unwraped_phase[i][j+1] - self.unwraped_phase[i][j])/(
                        subcarriers_index[j+1] -subcarriers_index[j])
                r_list.append(r)
            var_value = np.var(r_list)
            if var_value < self.value:
                fg_frame.append(self.unwraped_phase[i])
        return fg_frame

    def get_fingerprinting_list(self):
        """
        输入：归一化后的帧的子载波相位集合
        输出：将λ带入后的子载波相位集合
        """
        pi = math.pi
        frame_finger_list = list()
        for i in self.atf_lpf:
            fingerprint_list = list()
            z = (i[-1]+ i[0]) /2
            #z = (i[14]+ i[15]) /2
            k = (i[-1]- i[0]) / (112 * pi)
            #k = (i[14]- i[15]) / (4 * pi)
            for j in range(30):
                e = i[j] - (2 * pi * k *subcarriers_index[j]) - z
                fingerprint_list.append(e)
            frame_finger_list.append(fingerprint_list)
        return frame_finger_list

    def get_amplitude_list(self):
        """
        输入：子载波振幅集合
        输出：去除平均值的子载波振幅集合
        """
        frame_amplitude_list= list()
        for i in self.amplitude:
            amplitude_list = list()
            sum = 0
            for k in range(len(i)):
                sum += i[k]
            avg = sum / len(i)
            for j in range(30):
                e = i[j] - avg
                amplitude_list.append(e)
            frame_amplitude_list.append(amplitude_list)
        return frame_amplitude_list


    def average_fingerprint(self):
        return np.mean(self.finger_print,axis = 0)

    def average_amplitude(self):
        return np.mean(self.amplitude,axis = 0)

    def average_phase(self):
        return np.mean(self.unwraped_phase,axis = 0)

    def get_csi(self):
        print(self.csi_data)

    def get_raw_phase(self):
        print(self.raw_phase)

    def get_unwraped_phase(self):
        print(self.unwraped_phase)

    def get_origin_num(self):
        print(len(self.atf_lpf))

    def get_finger_num(self):
        print(len(self.finger_print))

    def get_data(self):
        data_sum = dict()
        data_sum['name'] = self.filename   #split('-')[-2]
        data_sum['csi'] = self.csi_data
        data_sum['rawpha'] = self.raw_phase
        data_sum['unwrappha'] = self.unwraped_phase
        data_sum['amt'] = self.amplitude
        data_sum['filterpha'] = self.atf_lpf
        data_sum['fplist'] = self.finger_print
        data_sum['fp'] = self.average_fingerprint()
        data_sum['amplitudef'] = self.average_amplitude()
        data_sum['aplist'] = self.amplitude_print
        return data_sum

def matching_cubic(fp_list):
    # 三次函数
    '''
    :param fp_list: 输入参数
    :return:
    '''
    x = np.array(subcarriers_index)
    y = np.array(fp_list)
    f1 = np.polyfit(x,y,3)
    p1 = np.poly1d(f1)
    a,b,c,d = p1[3],p1[2],p1[1],p1[0]
    print(p1)
    '''
    #yvals = p1(x)
    plot1 = plt.plot(x,y,'s',label = 'original values')
    plot2 = plt.plot(x, yvals,'r',label = 'polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-0.8,0.8)
    plt.legend(loc = 4)
    plt.title('polyfitting')
    plt.show()
    '''
    return [a,b,c,d]

def func(x,a,b,c,d):

    return a * np.sin(2*np.pi*b*x/56+c)+d

def matching_sin(fp_list):
    # asin(bx+c)+d拟合
    x = np.array(subcarriers_index)
    y = np.array(fp_list)

    popt,pcov = optimize.curve_fit(func,x,y)
    a = popt[0]
    b = popt[1]
    c = popt[2]
    d = popt[3]
    yvals = func(x,a,b,c,d)
    print(u'系数a：',a)
    print(u'系数b：',b)
    print(u'系数c：',c)
    print(u'系数d：',d)

    plot1 = plt.plot(x,y,'s',label = 'original values')
    plot2 = plt.plot(x,yvals,'r',label = 'polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-0.8,0.8)
    plt.legend(loc = 4)
    plt.title('curve_fit')
    plt.show()
    plt.savefig('test2.png')

def point_test(fp_list1,fp_list2):
    x = np.array(subcarriers_index)

    fp_list3 = [fp_list1,fp_list2]
    #fp_list3 = np.array(fp_list)
    make = cycle(['.', 'v', 'o', 's', '*', '+', '|', '^'])
    temp = 0
    for i in fp_list3:
        y = np.array(i)
        #plt.plot(subcarriers_index, y, marker=next(make))
        popt, pcov = optimize.curve_fit(func, x, y)
        tmp_a = popt[0]
        tmp_b = popt[1]
        tmp_c = popt[2]
        tmp_d = popt[3]
        yvals = func(x, tmp_a, tmp_b, tmp_c, tmp_d)
        #print(u'系数a：', tmp_a)
        #print(u'系数b：', tmp_b)
        #print(u'系数c：', tmp_c)
        #print(u'系数d：', tmp_d)
        if temp == 0:
            a1,b1,c1,d1 = tmp_a,tmp_b,tmp_c,tmp_d
            temp += 1

        plot = plt.plot(x, y, 's', label='original values')
        plot2 = plt.plot(x, yvals, 'r', label='polyfit values')

    print('系数a：', a1 ,'->', tmp_a)
    print('系数b：', b1, '->', tmp_b)
    print('系数c：', c1, '->', tmp_c)
    print('系数d：', d1, '->', tmp_d)

    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.show()

def set_fingerprint_database(csi_list):
    list = []
    for k in csi_list:
        list.append(k['fp'])
    x = np.array(subcarriers_index)
    for i in range(len(list)):
        y = np.array(list[i])
        popt, pcov = optimize.curve_fit(func, x, y)
        if i == 0:
            a_max = a_min = popt[0]
            b_max = b_min = popt[1]
            c_max = c_min = popt[2]
            d_max = d_min = popt[3]
        else:
            if popt[0] > a_max:
                a_max = popt[0]
            elif popt[0] < a_min:
                a_min = popt[0]
            if popt[1] > b_max:
                b_max = popt[1]
            elif popt[1] < b_min:
                b_min = popt[1]
            if popt[2] > c_max:
                c_max = popt[2]
            elif popt[2] < c_min:
                c_min = popt[2]
            if popt[3] > d_max:
                d_max = popt[3]
            elif popt[3] < d_min:
                d_min = popt[3]

        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
    '''
        yvals = func(x, a, b, c, d)
        #plot = plt.plot(x, y, 's', label='original values')
        #plot2 = plt.plot(x, yvals, 'r', label='polyfit values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.ylim(-1.5, 1.5)
    plt.show()
    '''
    print('a的最大值为：',a_max,'a的最小值为：',   a_min)
    print('b的最大值为：', b_max, 'b的最小值为：', b_min)
    print('c的最大值为：', c_max, 'c的最小值为：', c_min)
    print('d的最大值为：', d_max, 'd的最小值为：', d_min)
    print('---------------------------------------------')

    return a_min,a_max,b_min,b_max,c_min,c_max,d_min,d_max




def plot_img(csi_data):
    fig_rawphase = plt.subplot(3,3,1)
    plt.ylabel("Phase(radian)")
    plt.xlabel("Subcarriers Index")
    for i in csi_data['rawpha']:
        fig_rawphase.plot(subcarriers_index,i)
    fig_rawamti = plt.subplot(3,3,2)
    plt.ylabel("Amplitude")
    plt.xlabel("Subcarriers Index")
    plt.ylim(-10,50)
    for i in csi_data['amt']:
        fig_rawamti.plot(subcarriers_index,i)
    fig_unwrapphase = plt.subplot(3,3,3)
    plt.ylabel("Phase(radian")
    plt.xlabel("Subcarriers Index")
    for i in csi_data['unwrappha']:
        fig_unwrapphase.plot(subcarriers_index,i)
    fig_aftlpf = plt.subplot(3,3,4)
    plt.ylabel("Phase(raidian)")
    plt.xlabel("Subcarriers Index")
    for i in csi_data['filterpha']:
        #过滤后的子载波相位集合
        fig_aftlpf.plot(subcarriers_index,i)
    fig_fingerprint = plt.subplot(3,3,5)
    plt.ylabel("Phase(raidian)")
    plt.xlabel("Subcarriers Index")
    plt.ylim(-3,3)
    for i in csi_data['fplist']:
        #指纹
        fig_fingerprint.plot(subcarriers_index,i,'-')
    fig_averagefinger = plt.subplot(3,3,6)
    plt.ylabel("Phase(raidian)")
    plt.xlabel("Subcarriers Index")
    #指纹平均值
    fig_averagefinger.plot(subcarriers_index,csi_data['fp'],'-')
    plt.ylim(-3,3)
    fig_averageamplitude = plt.subplot(3, 3, 7)
    plt.ylabel("Amplitude")
    plt.xlabel("Subcarriers Index")
    # 指纹平均值
    fig_averageamplitude.plot(subcarriers_index, csi_data['amplitudef'], '-')
    plt.ylim(-10, 30)
    fig_Amplitudeprint = plt.subplot(3, 3, 8)
    plt.ylabel("Amplitude")
    plt.xlabel("Subcarriers Index")
    plt.ylim(-10, 50)

    for i in csi_data['aplist']:
        fig_Amplitudeprint.plot(subcarriers_index, i, '-')
    plt.show()

def get_max(plot_list):
    max_value = []
    min_value = []
    for j in range(30):
        temp = []
        for i in plot_list:
            temp.append(i['fp'][j])
        min_value.append(min(temp))
        max_value.append(max(temp))

    plt.xlabel("subcarriers index")
    plt.ylabel("phase error(radian)")
    make = cycle(['.', 'v', 'o', 's', '*', '+', '|', '^'])
    plt.plot(subcarriers_index, max_value, marker=next(make), linestyle='-', label=i['name'])
    plt.plot(subcarriers_index, min_value, marker=next(make), linestyle='-', label=i['name'])
    plt.ylim(-0.8, 0.8)
    plt.legend()
    plt.show()
    return max_value,min_value


def plot_single_img(plot_list1,plot_list2):
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.xlabel("subcarriers index",font2)
    plt.ylabel("phase error(radian)",font2)
    make = cycle(['.','v','o','s','*','+','|','^'])
    flag = 0
    for i in plot_list1:
        if flag == 0:
            plt.plot(subcarriers_index, i['fp'],c='r', linestyle='-',label='AP1')
            flag += 1
        else:
            plt.plot(subcarriers_index,i['fp'],c='r',linestyle='-')#marker= next(make),,label=i['name']
            flag += 1
    flag = 0
    for i in plot_list2:
        if flag == 0:
            plt.plot(subcarriers_index, i['fp'],c='cornflowerblue', linestyle='-',label='AP2')
            flag += 1
        else:
            plt.plot(subcarriers_index, i['fp'],c = 'cornflowerblue' ,linestyle='-')#marker=make,label=i['name']
            flag += 1


    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.tick_params(labelsize=16)
    plt.ylim(-1.5,1.5)
    plt.legend(bbox_to_anchor=(0.98, 0.02), loc=4, borderaxespad=0,prop=font1)
    plt.show()


def Recognition1(csi_list,fingerprint):
    for i in csi_list:
        flag_a,flag_b,flag_c,flag_d = 0,0,0,0
        x = np.array(subcarriers_index)
        y = np.array(i['fp'])
        popt, pcov = optimize.curve_fit(func, x, y)
        a = popt[0]
        b = popt[1]
        c = popt[2]
        d = popt[3]
        if a < fingerprint[0] or a > fingerprint[1]:
            flag_a = 1
        if b < fingerprint[2] or b > fingerprint[3]:
            flag_b = 1
        if c < fingerprint[4] or c > fingerprint[5]:
            flag_c = 1
        if d< fingerprint[6] or d > fingerprint[7]:
            flag_d = 1
        if flag_a == 1 or flag_b == 1 or flag_c == 1 or flag_d == 1:
            print("设备"+i['name']+'非法！非法参数：',flag_a,flag_b,flag_c,flag_d)
        else:
            print("设备"+i['name']+"合法设备")


def Recognition(csi_data,fingerprint):

    distance = 9999
    str =''
    for key in fingerprint:
        a = fingerprint[key]
        b = [csi_data['fp']]
        vec1 = np.array(a)
        vec2 = np.array(b)
        tmp_distance = np.sqrt(np.sum(np.square(vec1 - vec2)))
        if tmp_distance < distance:
            distance = tmp_distance
            str = key
    name = str.split('_',3)
    print('正在识别的设备为：'+ name[0]+ '-' +name[1])


def set_fingerprint(csi_list):

    fingerprint = {}
    plot_list = []
    for i in csi_list:
        csi = CSI(i, 0.006).get_data()
        fingerprint[csi['name']] = csi['fp']
        plot_list.append(csi)
    return fingerprint,plot_list

def get_area(func_value1,func_value2):
    x = symbols('x')
    f1 = func_value1[0]*x**3 - func_value1[1]*x**2 + func_value1[2]*x + func_value1[3]
    f2 = func_value2[0]*x**3 - func_value2[1]*x**2 + func_value2[2]*x + func_value2[3]
    func = f1 - f2
    print(integrate(func, (x, -28, 28)))

def max_append(list_sum):
    a1, a2, b1, b2, c1, c2, d1, d2 = [], [], [], [], [], [], [], []
    for list in list_sum:
        a1.append(list[0])
        a2.append(list[1])
        b1.append(list[2])
        b2.append(list[3])
        c1.append(list[4])
        c2.append(list[5])
        d1.append(list[6])
        d2.append(list[7])
    return a1, a2, b1, b2, c1, c2, d1, d2

def plot1(plot_list1):
    font2 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.xlabel("subcarriers index",font2)
    plt.ylabel("phase error(radian)",font2)
    make = cycle(['.','v','o','s','*','+','|','^'])
    for i in plot_list1:
        plt.plot(['1','2','3','4','5'],i,marker= next(make),linestyle='-')#,label=i['name']

    font1 = {'family': 'Times New Roman',
             'weight': 'normal',
             'size': 16,
             }
    plt.tick_params(labelsize=16)
    plt.ylim(-0.25,1.5)
    plt.legend(bbox_to_anchor=(0.98, 0.02), loc=4, borderaxespad=0,prop=font1)
    plt.show()

if __name__ == '__main__':

    csi_1 = CSI("UTT_12.27_1_10.dat", 0.06).get_data()
    csi_2 = CSI("UTT_1.1_1_8.dat", 0.005).get_data()
    csi_3 = CSI("UTT_1.1_1_3.dat", 0.005).get_data()
    csi_4 = CSI("UTT_1.1_1_4.dat", 0.005).get_data()
    csi_5 = CSI("UTT_1.1_1_5.dat", 0.005).get_data()
    csi_6 = CSI("UTT_1.1_1_6.dat", 0.005).get_data()
    csi_7 = CSI("UTT_1.1_1_7.dat", 0.005).get_data()
    csi_8 = CSI("UTT_1.1_1_8.dat", 0.005).get_data()
    csi_9 = CSI("UTT_1.1_1_9.dat", 0.005).get_data()
    csi_10 = CSI("UTT_1.1_1_10.dat", 0.005).get_data()

    csi_11 = CSI("FreeWiFi_12.31_1_1.dat", 0.005).get_data()
    csi_12 = CSI("FreeWiFi_12.31_1_2.dat", 0.005).get_data()
    csi_13 = CSI("FreeWiFi_12.31_1_3.dat", 0.005).get_data()
    csi_14 = CSI("FreeWiFi_12.31_1_3.dat", 0.005).get_data()
    csi_15 = CSI("FreeWiFi_12.31_1_5.dat", 0.005).get_data()
    csi_16 = CSI("FreeWiFi_12.31_1_6.dat", 0.005).get_data()
    csi_17 = CSI("FreeWiFi_12.31_1_7.dat", 0.005).get_data()
    csi_18 = CSI("FreeWiFi_12.31_1_8.dat", 0.005).get_data()
    csi_19 = CSI("FreeWiFi_12.31_1_9.dat", 0.005).get_data()
    csi_20 = CSI("FreeWiFi_12.31_1_10.dat", 0.005).get_data()

    csi_21 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_22 = CSI("FreeWiFi_1.3_1_2.dat", 0.005).get_data()
    csi_23 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_24 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_25 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_26 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_27 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_28 = CSI("FreeWiFi_1.3_1_4.dat", 0.005).get_data()
    csi_29 = CSI("FreeWiFi_1.3_1_9.dat", 0.005).get_data()
    csi_30 = CSI("FreeWiFi_1.3_1_10.dat", 0.005).get_data()

    print(csi_1)
    '''
    a_min, a_max, b_min, b_max, c_min, c_max, d_min, d_max = set_fingerprint_database(
        [csi_1, csi_2, csi_3, csi_4, csi_5, csi_6, csi_7, csi_8, csi_9, csi_10])
    a_min1, a_max1, b_min1, b_max1, c_min1, c_max1, d_min1, d_max1 = set_fingerprint_database(
        [csi_11,csi_12,csi_13,csi_14,csi_15,csi_16,csi_17,csi_18,csi_19])
    a_min2, a_max2, b_min2, b_max2, c_min2, c_max2, d_min2, d_max2 = set_fingerprint_database(
        [csi_21,csi_22,csi_23,csi_24,csi_25,csi_26,csi_27,csi_28,csi_29])
    a_min3, a_max3, b_min3, b_max3, c_min3, c_max3, d_min3, d_max3 = set_fingerprint_database(
        [csi_31, csi_32])
    a_min4, a_max4, b_min4, b_max4, c_min4, c_max4, d_min4, d_max4 = set_fingerprint_database(
        [csi_33, csi_34,csi_35])
        #[csi_1['fp'], csi_2['fp'],csi_3['fp'],csi_4['fp'],csi_5['fp'],csi_6['fp'],csi_7['fp'],csi_8['fp'],csi_9['fp'],csi_10['fp']])
    a1,a2,b1,b2,c1,c2,d1,d2 = max_append([[a_min, a_max, b_min, b_max, c_min, c_max, d_min, d_max],
                                          [a_min1, a_max1, b_min1, b_max1, c_min1, c_max1, d_min1, d_max1],
                                          [a_min2, a_max2, b_min2, b_max2, c_min2, c_max2, d_min2, d_max2],
                                          [a_min3, a_max3, b_min3, b_max3, c_min3, c_max3, d_min3, d_max3],
                                          [a_min4, a_max4, b_min4, b_max4, c_min4, c_max4, d_min4, d_max4]])
    '''

    #fingerprint_databese = [a_min, a_max, b_min, b_max, c_min, c_max, d_min, d_max]

    #recognition_databese = [csi_4, csi_5,csi_6]
    #max_value,min_value = get_max([csi_1,csi_2,csi_3,csi_5,csi_6,csi_7,csi_8,csi_9,csi_10])
    #matching_sin(max_value)
    #matching_sin(min_value)
    #max_func = matching_cubic(max_value)
   # min_func = matching_cubic(min_value)
   # print(max_func,min_func)
    #get_area(max_func,min_func)
    #get_area(max_func)
                    #[csi_11,csi_12,csi_13,csi_14,csi_15,csi_16,csi_17,csi_18,csi_19])




    #a_min,a_max,b_min,b_max,c_min,c_max,d_min,d_max = set_fingerprint_database([csi_1['fp'],csi_2['fp'],csi_3['fp'],csi_4['fp'],csi_5['fp'],
    #                          csi_6['fp'],csi_7['fp'],csi_8['fp'],csi_9['fp'],csi_10['fp']])

    #set_fingerprint_database([csi_1['fp'],csi_2['fp'],csi_19['fp']])

    #set_fingerprint_database([csi_11['fp'], csi_12['fp'], csi_13['fp'], csi_14['fp'], csi_15['fp'],
    #                       csi_16['fp'], csi_17['fp'], csi_18['fp']])













