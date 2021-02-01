from IT8951 import constants
from PIL import Image, ImageOps

class Display():
    display = None
    width = 0
    height = 0
    
    def init_display(self, virtual=False):
        print('[INFO] Initializing E-Paper Display')
        # Rotation: 'None', 'CW', 'CCW', 'flip'
        if virtual:
            from IT8951.display import VirtualEPDDisplay
            self.display = VirtualEPDDisplay(dims=(1872, 1404), rotate=None)
        else:
            from IT8951.display import AutoEPDDisplay
            self.display = AutoEPDDisplay(vcom=-2.06, rotate=None, spi_hz=24000000)
            print('[INFO] VCOM set to', self.display.epd.get_vcom())
            
        self.width = self.display.epd.width
        self.height = self.display.epd.height
        
    def print_system_info(self):
        epd = self.display.epd
        print('[INFO] Display info:')
        print('  display size: {}x{}'.format(self.display.epd.width, self.display.epd.height))
        print('  img buffer address: {:X}'.format(self.display.epd.img_buf_address))
        print('  firmware version: {}'.format(self.display.epd.firmware_version))
        print('  LUT version: {}'.format(self.display.epd.lut_version))

    def clear(self):
        print('[INFO] Clearing display...')
        self.display.clear()

    def display_image(self, image):
        print('[INFO] Setting new image to display...')
        self.display.frame_buf.paste(0xFF, box=(0, 0, self.display.width, self.display.height))
        dims = (self.display.width, self.display.height)
        image.thumbnail(dims)
        self.display.frame_buf.paste(image, (0, 0, image.size[0], image.size[1]))
        self.display.draw_full(constants.DisplayModes.GC16)
