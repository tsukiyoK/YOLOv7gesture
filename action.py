import win32gui
import win32con
import win32api
import time
import configparser


def volumeUp():
    n = 5   #volumeDefault
    while n:
        win32api.keybd_event(win32con.VK_VOLUME_UP, 0)
        win32api.keybd_event(win32con.VK_VOLUME_UP, 0, win32con.KEYEVENTF_KEYUP)
        n = n - 1
    return


def volumeDown():
    n = 5   #volumeDefault
    while n:
        win32api.keybd_event(win32con.VK_VOLUME_DOWN, 0)
        win32api.keybd_event(win32con.VK_VOLUME_DOWN, 0, win32con.KEYEVENTF_KEYUP)
        n = n - 1
    return


def mediaPause():
    win32api.keybd_event(179, 0)
    win32api.keybd_event(179, 0, win32con.KEYEVENTF_KEYUP)


def mediaNextTrack():
    win32api.keybd_event(win32con.VK_MEDIA_NEXT_TRACK, 0)
    win32api.keybd_event(win32con.VK_MEDIA_NEXT_TRACK, 0, win32con.KEYEVENTF_KEYUP)
    return


def mediaPrevTrack():
    win32api.keybd_event(win32con.VK_MEDIA_PREV_TRACK, 0)
    win32api.keybd_event(win32con.VK_MEDIA_PREV_TRACK, 0, win32con.KEYEVENTF_KEYUP)


def volumeMute():
    win32api.keybd_event(173, 0)
    win32api.keybd_event(173, 0, win32con.KEYEVENTF_KEYUP)


# Mouse

def mouseLeftHold():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0)


def mouseLeftClk():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTDOWN, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0)


def mouseRightClk():
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTDOWN, 0, 0, 0)
    win32api.mouse_event(win32con.MOUSEEVENTF_RIGHTUP, 0, 0, 0)


def mouseReset():
    win32api.mouse_event(win32con.MOUSEEVENTF_LEFTUP, 0, 0, 0)

