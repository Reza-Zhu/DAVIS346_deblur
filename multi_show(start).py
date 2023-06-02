import queue
import numpy as np
import cv2
import threading
from copy import deepcopy

# dv_processing@https://dv-processing.inivation.com/rel_1.7/installation.html
import dv_processing as dv
import argparse
import time
import datetime
from PIL import Image
from inference import davis346_inference
from event_utils import hot_pixel_detect
from DAVIS_event_abnormal_detect import Event_package_loss

from model import EFNet
from model import load_network


## 采集DAVIS的子线程
thread_lock = threading.Lock()
thread_exit = False
deblur_waiting_flag = True
exposure_time = 40  ##ms
model = None


class DAVIS(threading.Thread):
    def __init__(self, img_height, img_width, cam):
        super(DAVIS, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.cam = cam
        # frame是一个常变的变量，deblur线程不会直接访问它。主线程会访问它，以显示DAVIS实时采集到的APS帧
        self.frame = np.zeros((img_height, img_width), dtype=np.uint8)
        self.event = None
        self.frame_timestamp = 0
        self.last_frame = np.zeros((img_height, img_width), dtype=np.uint8)
        self.last_frame_timestamp = 0
        # event_list用于存储一个较长时间段的事件序列
        self.event_list = None
        # deblur_events是event_list经过挑选后，符合送入deblur处理线程的事件序列
        self.deblur_events = None
        self.frame_enable = True
        self.exposure_start = 0
        self.exposure_end = 0
        self.debug_ts = 0

        self.acquire_flag = True


    def get_frame(self):
        return deepcopy(self.frame)

    def get_event(self):
        if self.event is not None:
            return self.event.numpy()

    def get_last_frame_timestamp(self):
        return self.last_frame_timestamp

    def get_last_frame(self):
        return deepcopy(self.last_frame)

    def get_event_list(self):
        return deepcopy(self.event_list)



    def get_deblur_events(self):
        if self.deblur_events is not None:
            return self.deblur_events
        else:
            return None

    def run(self):
        global thread_exit
        global deblur_waiting_flag
        while (not thread_exit and self.cam.isRunning()):
            # frame = cv2.resize(frame, (self.img_width, self.img_height))
            # 从设备buffer读取frame及event
            frame = self.cam.getNextFrame()

            events = self.cam.getNextEventBatch()

            thread_lock.acquire()
            if frame is not None:
                self.frame = frame.image
                self.frame_timestamp = frame.timestamp
                # time.sleep(1)

            if events is not None:
                self.event = events
                # event_bag=events.numpy().tolist()

            # deblur 用不到external trigger
            # if triggers is not None:
            #     # self.trigger_list = triggers[0:-1]
            #     #
            #     # print(f"Received imu data within time range [{triggers[0].timestamp}; {triggers[-1].timestamp}]")
            #     for i,trigger in enumerate(triggers):
            #         # print(trigger.timestamp)
            #         if trigger.type.name == 'APS_EXPOSURE_START':
            #             self.exposure_start = trigger.timestamp
            #         elif trigger.type.name == 'APS_EXPOSURE_END':
            #             self.exposure_end = trigger.timestamp
            #             # print(self.exposure_end-self.exposure_start)


            # 如果deblur线程空闲，
            if deblur_waiting_flag and (events is not None) and (frame is not None):
                # 将本次采集到的APS帧放入last_frame中
                if self.frame_enable:
                    self.last_frame_timestamp = self.frame_timestamp
                    self.last_frame = self.frame
                    self.frame_enable = False


                if self.event_list is None:
                    self.event_list = np.array([events.numpy().tolist()]).squeeze()
                else:
                    input_events = np.array([events.numpy().tolist()]).squeeze()

                    if self.debug_ts == 0:
                        self.debug_ts = input_events[-1,0]
                    else:
                        #TODO ：出现报错时候，把event_list清空
                        # if input_events[0, 0] - self.debug_ts > 20*exposure_time*1e3:
                        #     print('lss')
                        #     # self.event_list = None
                        #     # self.deblur_events = None
                        #     self.frame_enable = True
                        #     # time.sleep(0.0001)
                        #     thread_lock.release()
                        #     continue

                        self.debug_ts = input_events[-1,0]
                    self.event_list = np.append(self.event_list, input_events, axis=0)
                    #防止event_list中事件过多导致内存泄漏
                    if self.event_list.shape[0] > 330e3*10:
                        print('clear')
                        self.event_list = self.event_list[-int(330e3*5):,:]
                # 如果当前存储的event_list中开始事件的时间戳落后于当前存储的APS帧图像的时间戳，说明事件曝光时间内的事件不完整，丢弃这一帧，事件依然保留
                # 6.1添加：如果经过20个曝光时间，last_frame依然没有推理时，重新采一张frame存储下来（跟上实时显示的效果）
                if self.last_frame_timestamp < self.event_list[0, 0] or self.frame_timestamp-self.last_frame_timestamp > exposure_time*1e3*20:
                    # self.event_list = None  #event_list不清空
                    self.frame_enable = True  #即在下一次进入该线程时把新获取到的一帧APS存储下来
                elif self.last_frame_timestamp + exposure_time * 1000 < self.event_list[-1, 0]:
                    # 说明event_list已经包含last_frame曝光时间内的所有事件了
                    # print('deblur ready')
                    # start=time.time()

                    # 把event_list里曝光时间内的事件挑出来
                    low_index = np.argmin(np.abs(self.event_list[:, 0].squeeze() - self.last_frame_timestamp))
                    high_index = np.argmin(
                        np.abs(self.event_list[:, 0].squeeze() - self.last_frame_timestamp - exposure_time * 1000))
                    self.deblur_events = self.event_list[low_index:high_index, :].squeeze()

                    # deblur_waiting_flag = False指的是deblur线程可以开始进行deblur了
                    deblur_waiting_flag = False
                    self.event_list = self.event_list[high_index:, :].squeeze()
                    # 可以将接下来的APS流中的帧存储下来了
                    self.frame_enable = True
                    # end = time.time()
                    # print((end-start)*1000)
            thread_lock.release()


# 处理画面的子线程
class RenderFrame(threading.Thread):
    def __init__(self, thread_name, raw_thread, img_height, img_width):
        threading.Thread.__init__(self, name=thread_name)
        self.raw_thread = raw_thread
        self.input_frame = np.zeros((img_height, img_width), dtype='uint8')
        self.event_frame = np.zeros((img_height, img_width, 3), dtype='float')
        self.deblur_frame = np.zeros((img_height, img_width), dtype='uint8')
        self.input_frame_2 = np.zeros((img_height, img_width, 3), dtype='uint8')
        self.event_frame_2 = np.zeros((img_height, img_width, 3), dtype='float')
        self.hot_pixel_enum_list = np.zeros((260, 346),dtype='uint8')
    def get_input_frame(self):
        return deepcopy(self.input_frame)

    def get_input_frame_2(self):
        return deepcopy(self.input_frame_2)

    def get_deblur_frame(self):
        return deepcopy(self.deblur_frame)

    def get_event_frame(self):
        # self.event_frame = self.event_frame/np.max(self.event_frame)*255

        # return_event_frame = return_event_frame/np.max(return_event_frame)*255
        return deepcopy(self.event_frame)

    def get_event_frame_2(self):
        # self.event_frame = self.event_frame/np.max(self.event_frame)*255

        # return_event_frame = return_event_frame/np.max(return_event_frame)*255
        return deepcopy(self.event_frame_2)

    def run(self) -> None:
        global deblur_waiting_flag
        global model
        while (not thread_exit):
            thread_lock.acquire()
            frame = self.raw_thread.get_last_frame()
            last_timestamp = self.raw_thread.get_last_frame_timestamp()
            deblur_events = self.raw_thread.get_deblur_events()
            thread_lock.release()
            # deblur事件还不够
            if (deblur_events is None) or (deblur_events.size == 0):
                deblur_waiting_flag = True
                continue
            # 检测是否丢包，或者运动幅度不够大，没有足够的事件数量
            if Event_package_loss(deblur_events, exposure_time) or deblur_events.shape[0] < 8000:
                # event_list = self.raw_thread.get_event_list()
                deblur_waiting_flag = True
                continue
            # 采集线程已经采集到足够deblur的事件以及对应的APS帧了
            if not deblur_waiting_flag:
                # print(deblur_events.size)
                deblur_waiting_flag = True
                self.input_frame = frame
                if len(deblur_events.shape) > 1:
                    self.event_frame = np.zeros(self.event_frame.shape, dtype='float')
                else:
                    deblur_waiting_flag = True
                    continue

                # # 生成hot_pixel_detect使用的事件帧
                # for i in range(deblur_events.shape[0]):
                #     if deblur_events[i, 3] == 1:
                #         self.event_frame[deblur_events[i, 2], deblur_events[i, 1], 2] = self.event_frame[
                #                                                                             deblur_events[i, 2],
                #                                                                             deblur_events[
                #                                                                                 i, 1], 2] + 1
                #     else:
                #         self.event_frame[deblur_events[i, 2], deblur_events[i, 1], 0] = self.event_frame[
                #                                                                             deblur_events[i, 2],
                #                                                                             deblur_events[
                #                                                                                 i, 1], 0] + 1
                # hot_pixel_list = hot_pixel_detect(self.event_frame[:, :, 2].squeeze(),
                #                                   self.event_frame[:, :, 0].squeeze(), 4)
                # print(hot_pixel_list)



                # 去除hot pixel

                # 其实对于某个设备，每个设定bias，hot pixel应该是固定的，在默认阈值下，hot pixel 应该为
                # [[ 61 161]
                #  [136 197]
                #  [207 188]
                #  [237 257]]
                #
                #  如果从可视化事件帧里发现事件帧不对，可能是hot pixel 变了，这时候再uncomment上面的hot_pixel_detect
                hot_pixel_list = np.array([[61, 161], [136, 197], [207, 188], [237, 257]])
                print(f'Event count before hot pixel removal:{deblur_events.shape[0]}')

                for i in range(hot_pixel_list.shape[0]):
                    loc = np.where(
                        (deblur_events[:, 2] == hot_pixel_list[i, 0]) & (deblur_events[:, 1] == hot_pixel_list[i, 1]))
                    deblur_events = np.delete(deblur_events, loc, axis=0)




                if len(deblur_events.shape) > 1:
                    self.event_frame = np.zeros(self.event_frame.shape, dtype='float')
                    for i in range(deblur_events.shape[0]):
                        if deblur_events[i, 3] == 1:
                            self.event_frame[deblur_events[i, 2], deblur_events[i, 1], 2] = self.event_frame[
                                                                                                deblur_events[i, 2],
                                                                                                deblur_events[
                                                                                                    i, 1], 2] + 1
                        else:
                            self.event_frame[deblur_events[i, 2], deblur_events[i, 1], 0] = self.event_frame[
                                                                                                deblur_events[i, 2],
                                                                                                deblur_events[
                                                                                                    i, 1], 0] + 1

                # for i in range(hot_pixel_list.shape[0]):
                #     loc = np.where(
                #         (deblur_events[:, 2] == hot_pixel_list[i, 0]) & (deblur_events[:, 1] == hot_pixel_list[i, 1]))
                #     deblur_events = np.delete(deblur_events, loc, axis=0)

                print(f'Event count after hot pixel removal:{deblur_events.shape[0]}')
                cv2.normalize(self.event_frame, self.event_frame, 0, 1, cv2.NORM_MINMAX)
                self.event_frame = cv2.putText(self.event_frame, 'event num: %d' % deblur_events.shape[0], [0, 60],
                                               cv2.FONT_HERSHEY_SIMPLEX, 1, (1, 1, 1), 3)

                # 送入推理
                out_img = davis346_inference(frame, deblur_events, last_timestamp,model)
                self.deblur_frame = out_img
                self.event_frame_2 = self.event_frame
                self.input_frame_2[:, :, 0] = self.input_frame
                self.input_frame_2[:, :, 1] = self.input_frame
                self.input_frame_2[:, :, 2] = self.input_frame

                # time.sleep(0.5)
            # self.render_data.put(deblured_img)


def main():
    global thread_exit
    global deblur_waiting_flag
    global exposure_time
    global model
    parser = argparse.ArgumentParser(description='Show a preview of an iniVation event camera input.')
    args = parser.parse_args()
    cv2.namedWindow("DAVIS Frame Output", cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Input Frame', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Deblur Frame', cv2.WINDOW_GUI_NORMAL)
    cv2.namedWindow('Event Frame', cv2.WINDOW_GUI_NORMAL)
    # Open the camera
    camera = dv.io.CameraCapture()
    camera.setDavisExposureDuration(datetime.timedelta(milliseconds=exposure_time))
    camera.setDavisFrameInterval(datetime.timedelta(milliseconds=int(exposure_time)))

    # camera.setDVSGlobalHold(False)   # no use
    camera.deviceConfigSet(-3,1,50000)  #设定发包间隔，默认为10000微秒。改为50,000微妙后显示帧率降低，但是减少出现event buffer overflow的概率
    # 可能的修改阈值方法@https://gitlab.com/inivation/inivation-docs/-/blob/master/Advanced%20configurations/User_guide_-_Biasing.md
    # camera.deviceConfigSet(5, 12, 16420)
    # camera.deviceConfigSet(5, 11, 24573)
    print(camera.deviceConfigGet(-3, 1))
    # test = camera.caerDeviceConfigGet(camera.deviceConfigGet(5, 12))
    # 加载模型

    # weight_path = './net_g_latest_REBlur.pth'
    weight_path = './net_g_latest_GoPro.pth'
    # weight_path = './net_g_latest.pth'
    # weight_path = './net_g_gray.pth'
    model = EFNet().cuda()
    # 测试跑通不需要权重
    # model.load_state_dict(torch.load(weight_path))
    # TODO load_network 放到main 线程(出现过bug)
    model = load_network(model, weight_path, True, param_key='params')

    model.eval()

    # print(camera.isRunning())
    img_height = 260
    img_width = 346
    thread_davis = DAVIS(img_height, img_width, camera)
    thread_davis.start()
    deblur = RenderFrame("RenderFrame", thread_davis, img_height, img_width)
    deblur.start()  # 开始线程

    # 在主线程处理渲染完成的画面
    '''
    由于 OpenCV 的限制，无法在子线程中使用 nameWindow 或者 imshow 等方法
    只能新建一个多线程列队将渲染完成的信息加入到列队，然后再在主线程中展示出来
    '''
    while not thread_exit:
        # start_time = time.time()
        thread_exit = not camera.isRunning()
        thread_lock.acquire()
        frame = thread_davis.get_frame()

        deblur_frame = deblur.get_deblur_frame()
        render_frame = deblur.get_input_frame_2()
        event_frame = deblur.get_event_frame_2()
        thread_lock.release()
        cv2.imshow('DAVIS Frame Output', frame)

        # 如果获取的帧不为空
        cv2.imshow('Input Frame', render_frame)
        cv2.imshow('Event Frame', event_frame)
        cv2.imshow('Deblur Frame', deblur_frame)
        if cv2.waitKey(1) == 27:
            thread_exit = True
            cv2.destroyAllWindows()
        # end_time = time.time()
        # print("FPS: ", 1 / (end_time - start_time))
        # break
    # DAVIS346采集线程
    thread_davis.join()
    # Deblur 处理线程
    deblur.join()


if __name__ == '__main__':
    main()
