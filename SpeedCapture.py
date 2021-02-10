import ctypes
from ctypes import wintypes

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)

ERROR_PARTIAL_COPY = 0x012B
PROCESS_VM_READ = 0x0010

SIZE_T = ctypes.c_size_t
PSIZE_T = ctypes.POINTER(SIZE_T)


def _check_zero(result, func, args):
    if not result:
        raise ctypes.WinError(ctypes.get_last_error())
    return args


kernel32.OpenProcess.errcheck = _check_zero
kernel32.OpenProcess.restype = wintypes.HANDLE
kernel32.OpenProcess.argtypes = (
    wintypes.DWORD,  # _In_ dwDesiredAccess
    wintypes.BOOL,   # _In_ bInheritHandle
    wintypes.DWORD)  # _In_ dwProcessId

kernel32.ReadProcessMemory.errcheck = _check_zero
kernel32.ReadProcessMemory.argtypes = (
    wintypes.HANDLE,   # _In_  hProcess
    wintypes.LPCVOID,  # _In_  lpBaseAddress
    wintypes.LPVOID,   # _Out_ lpBuffer
    SIZE_T,            # _In_  nSize
    PSIZE_T)           # _Out_ lpNumberOfBytesRead

kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)


def read_process_memory(pid, address, size, allow_partial=False):
    buf = (ctypes.c_char * size)()
    nread = SIZE_T()
    hProcess = kernel32.OpenProcess(PROCESS_VM_READ, False, pid)
    try:
        kernel32.ReadProcessMemory(hProcess, address, buf, size,
                                   ctypes.byref(nread))
    except WindowsError as e:
        if not allow_partial or e.winerror != ERROR_PARTIAL_COPY:
            raise
    finally:
        kernel32.CloseHandle(hProcess)
    return buf[:nread.value]


def GetSpeed(PID, address, size=3, endian='little'):
    return int.from_bytes(
        read_process_memory(PID, address, size),
        endian
    )
