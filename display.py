from IT8951 import constants
from PIL import Image, ImageOps
import logging

logging.basicConfig(format='[%(asctime)s] [%(levelname)s] %(message)s', level=logging.INFO)

class Display():
    display_device = None
    width = 0
    height = 0
    main_picture = None
    cam_picture = None
    img_border = 5
    
    def init_display(self, virtual=False):
        logging.info('Initializing E-Paper Display')
        rotation = None
        # Rotation: 'None', 'CW', 'CCW', 'flip'
        if virtual:
            from IT8951.display import VirtualEPDDisplay
            self.display_device = VirtualEPDDisplay(dims=(1872, 1404), rotate=rotation)
        else:
            from IT8951.display import AutoEPDDisplay
            self.display_device = AutoEPDDisplay(vcom=-1.56, rotate=rotation, spi_hz=24000000)
            logging.info('VCOM set to' + str(self.display_device.epd.get_vcom()))
        
        if rotation == 'CW' or rotation == 'CCW':
            self.height = self.display_device.epd.width
            self.width = self.display_device.epd.height
        else:
            self.width = self.display_device.epd.width
            self.height = self.display_device.epd.height
        
    def print_system_info(self):
        epd = self.display_device.epd
        logging.info('Display info:')
        logging.info('  display size: {}x{}'.format(epd.width, epd.height))
        logging.info('  img buffer address: {:X}'.format(epd.img_buf_address))
        logging.info('  firmware version: {}'.format(epd.firmware_version))
        logging.info('  LUT version: {}'.format(epd.lut_version))

    def clear(self):
        logging.info('Clearing display...')
        self.display_device.clear()

    def draw_image(self, image, position, halign='center', valign='center'):
        x = position[0] + self.img_border
        y = position[1] + self.img_border
        if halign == 'center':
            x += int((position[2]-position[0]-image.width)/2)
        elif halign == 'left':
            x = position[0]+self.img_border
        elif halign == 'right':
            x = position[2]-image.width+self.img_border

        if valign == 'center':
            y += int((position[3]-position[1]-image.height)/2)
        elif valign == 'top':
            y = position[1]+self.img_border
        elif valign == 'bottom':
            y = position[3]-image.height+self.img_border
            
        w = x + image.width
        h = y + image.height
        self.display_device.frame_buf.paste(image, (x, y, w, h))
        
    def update(self):
        logging.info('Updating display...')
        self.display_device.frame_buf.paste(0xFF, box=(0, 0, self.width, self.height))
        if self.main_picture is not None:
            self.draw_image(self.main_picture, [320, 0, self.width, self.height])
        if self.cam_picture is not None:
            self.draw_image(self.cam_picture, [0, 0, 320, 240])
        self.display_device.draw_partial(constants.DisplayModes.GC16)

    def set_cam_image(self, image):
        logging.info('Setting camera picture to display...')
        if image is not None:
            image.thumbnail((320-self.img_border*2, 240-self.img_border*2))
            image = image.convert("L")
            image = image.point(lambda x: int(x/17)*17)
            self.cam_picture = image.copy()
        
    def set_main_image(self, image):
        logging.info('Setting main picture to display...')
        if image is not None:
            image.thumbnail((self.width-320-self.img_border*2, self.height-self.img_border*2))
            image = image.convert("L")
            image = image.point(lambda x: int(x/17)*17)
            self.main_picture = image.copy()