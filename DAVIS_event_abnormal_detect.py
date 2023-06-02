import numpy as np

# 检测input_events是否包含曝光时间内足够的事件
def Event_package_loss(input_event,expousure_time,thd=0.3):
    if input_event[-1,0]-input_event[0,0]<expousure_time*1e3*thd:
        print(f'Warning: Events Packet Loss!')
        return True

#TODO：检测input_event中是否发生了事件时间戳断层。事件时间戳断层是由于event_list积累过程中，发生了设备事件缓存区溢出，导致丢失了中间一部分事件。
def Event_buffer_overflow(input_event):
    pass