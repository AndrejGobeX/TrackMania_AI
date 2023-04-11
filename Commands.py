import vgamepad as vg
from DirectKey import PressKey, ReleaseKey, KEY_DELETE, KEY_UP, KEY_DOWN, KEY_RIGHT, KEY_LEFT, KEY_ENTER
import time

gamepad = vg.VX360Gamepad()

# Steer - left_joystick_float
# Gas - right_trigger_float
# Brake - left_trigger_float


def tm_respawn():
    PressKey(KEY_DELETE)
    ReleaseKey(KEY_DELETE)
    time.sleep(0.1)
    PressKey(KEY_ENTER)
    ReleaseKey(KEY_ENTER)
    time.sleep(0.1)
    PressKey(KEY_DELETE)
    ReleaseKey(KEY_DELETE)


def tm_accelerate(v):
    gamepad.right_trigger_float(value_float=v)


def tm_accelerate_keyboard(v):
    if v > 0.5:
        PressKey(KEY_UP)
    else:
        ReleaseKey(KEY_UP)


def tm_brake(v):
    gamepad.left_trigger_float(value_float=v)
    

def tm_brake_keyboard(v):
    if v > 0.5:
        PressKey(KEY_DOWN)
    else:
        ReleaseKey(KEY_DOWN)


def tm_steer(v):
    gamepad.left_joystick_float(x_value_float=v, y_value_float=0)


def tm_steer_keyboard(v):
    if v > 0.5:
        ReleaseKey(KEY_LEFT)
        PressKey(KEY_RIGHT)
    elif v < -0.5:
        ReleaseKey(KEY_RIGHT)
        PressKey(KEY_LEFT)
    else:
        ReleaseKey(KEY_LEFT)
        ReleaseKey(KEY_RIGHT)


def tm_update():
    gamepad.update()


def tm_reset():
    gamepad.reset()


def tm_reset_keyboard():
    ReleaseKey(KEY_UP)
    ReleaseKey(KEY_DOWN)
    ReleaseKey(KEY_RIGHT)
    ReleaseKey(KEY_LEFT)

