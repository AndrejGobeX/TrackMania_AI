from time import sleep
import vgamepad as vg
from DirectKey import PressKey, ReleaseKey, KEY_DELETE, KEY_UP, KEY_DOWN

gamepad = vg.VX360Gamepad()

# Steer - left_joystick_float
# Gas - right_trigger_float
# Brake - left_trigger_float


def tm_respawn():
    PressKey(KEY_DELETE)
    ReleaseKey(KEY_DELETE)


def tm_accelerate(v):
    #gamepad.right_trigger_float(value_float=v)
    if v > 0.5:
        PressKey(KEY_UP)
    else:
        ReleaseKey(KEY_UP)


def tm_brake(v):
    #gamepad.left_trigger_float(value_float=v)
    if v > 0.5:
        PressKey(KEY_DOWN)
    else:
        ReleaseKey(KEY_DOWN)


def tm_steer(v):
    gamepad.left_joystick_float(x_value_float=v, y_value_float=0)


def tm_update():
    gamepad.update()


def tm_reset():
    gamepad.reset()
    ReleaseKey(KEY_UP)
    ReleaseKey(KEY_DOWN)

