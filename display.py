from IT8951 import constants
from PIL import Image, ImageFont, ImageOps, ImageDraw
import logging
import datetime
from random import randint
import math
import calendar

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
        border = 20

        im = Image.new('L', (self.width, self.height), 10*16) 
        draw = ImageDraw.Draw(im) 

        draw.rectangle((border,border,self.width-border,self.height-border), fill=15*16, outline=0) 
        draw.rectangle((border,border,360+border,self.height-border), fill=15*16, outline=0) 

        self.draw_date_calendar(draw, (border*4, border*4, 320, 320))
        self.draw_clock(draw, (border*3, 320+border*3, 320, 640))
        self.draw_month_calendar(draw, (border*2, 710+border*4, 320+border*2, 1100))

        self.display_device.frame_buf.paste(im, box=(0, 0, self.width, self.height))
        if self.main_picture is not None:
            self.draw_image(self.main_picture, [360+border*3, border*4, self.width-border*3, self.height-border*4])
        if self.cam_picture is not None:
            self.draw_image(self.cam_picture, [border, self.height-270-border, 360+border, self.height-border])
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
            image = ImageOps.fit(image, (self.width-360-20*4, self.height-20*4))
            image = image.convert("L")
            image = image.point(lambda x: int(x/17)*17)
            self.main_picture = image.copy()
            
    def rounded_rectangle(self, draw, xy, radius, fill=None, outline=None, corner=[True, True, True, True]):
        (x1, y1, x2, y2) = xy
        # Center
        draw.rectangle([
            x1 + radius, y1 + radius, 
            x2 - radius, y2 - radius
            ], fill=fill, outline=outline)
        # Top
        draw.rectangle([
            x1 + (radius if corner[0] else 0), y1, 
            x2 - (radius if corner[1] else 0), y1 + radius
            ], fill=fill, outline=outline)
        # Bottom
        draw.rectangle([
            x1 + (radius if corner[3] else 0), y2 - radius, 
            x2 - (radius if corner[2] else 0), y2
            ], fill=fill, outline=outline)
        # Left
        draw.rectangle([
            x1, y1 + (radius if corner[0] else 0), 
            x1 + radius, y2 - (radius if corner[3] else 0)
            ], fill=fill, outline=outline)
        # Right
        draw.rectangle([
            x2 - radius, y1 + (radius if corner[1] else 0), 
            x2, y2 - (radius if corner[2] else 0)
            ], fill=fill, outline=outline)
        # Top Left
        if corner[0]:
            draw.pieslice([
                x1, y1, 
                x1 + radius*2, y1 + radius*2],
                180, 270, fill=fill, outline=outline
            )
        # Top Right
        if corner[1]:
            draw.pieslice([
                x2 - radius*2, y1,
                x2, y1 + radius*2],
                270, 360, fill=fill, outline=outline
            )
        # Bottom Right
        if corner[2]:
            draw.pieslice([
                x2 - radius*2, y2 - radius*2,
                x2, y2],
                0, 90, fill=fill, outline=outline
            )
        # Bottom Left
        if corner[3]:
            draw.pieslice([
                x1, y2 - radius*2,
                x1 + radius*2, y2],
                90, 180, fill=fill, outline=outline
            )

    def draw_date_calendar(self, draw, xy, random=False):
        x = datetime.datetime.now()
        if random:
            x = datetime.date(randint(2005,2025), randint(1,12),randint(1,28))
        dat = x.strftime("%d")
        month = x.strftime("%B")
        day = x.strftime("%A")

        (x1, y1, x2, y2) = xy
        radius = int((x2-x1)/7)
        # Shadows
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/2, x2, y2+8), radius, fill=11*16, corner=[False, False, True, True])
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/2, x2, y2+4), radius, fill=9*16, corner=[False, False, True, True])
        # Top 1/4
        self.rounded_rectangle(draw, (x1, y1, x2, y1+(y2-y1)/2), radius, fill=9*16, corner=[True, True, False, False])
        # Bottom 3/4
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/4, x2, y2), radius, fill=13*16, corner=[False, False, True, True])
        
        font = ImageFont.truetype('./Open_Sans/OpenSans-SemiBold.ttf', radius)
        draw.text(((x1+x2)/2, y1+35), month, font=font, anchor="mm", fill=16*16)
        draw.text(((x1+x2)/2, y2-35), day, font=font, anchor="mm", fill=7*16)
        font = ImageFont.truetype('./Open_Sans/OpenSans-SemiBold.ttf', radius*4)
        draw.text(((x1+x2)/2, (y1+y2)/2), dat, font=font, anchor="mm", fill=4*16)

    def draw_month_calendar(self, draw, xy, random=False):
        x = datetime.datetime.now()
        if random:
            x = datetime.date(randint(2005,2025), randint(1,12), randint(1,28))
        dat = x.strftime("%d")
        month = x.strftime("%B")
        day = x.strftime("%A")

        (x1, y1, x2, y2) = xy
        radius = int((x2-x1)/12)
        # Shadows
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/2, x2, y2+8), radius, fill=11*16, corner=[False, False, True, True])
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/2, x2, y2+4), radius, fill=9*16, corner=[False, False, True, True])
        # Top 1/4
        self.rounded_rectangle(draw, (x1, y1, x2, y1+(y2-y1)/5), radius, fill=9*16, corner=[True, True, False, False])
        # Bottom 3/4
        self.rounded_rectangle(draw, (x1, y1+(y2-y1)/5, x2, y2), radius, fill=13*16, corner=[False, False, True, True])
        
        font = ImageFont.truetype('./Open_Sans/OpenSans-SemiBold.ttf', int(radius*1.25))
        draw.text(((x1+x2)/2, y1+35), x.strftime("%B %Y"), font=font, anchor="mm", fill=14*16)
        font = ImageFont.truetype('./Open_Sans/OpenSans-SemiBold.ttf', int(radius*0.9))
        days = ['Mo', 'Tu', 'We', 'Th', 'Fr', 'Sa', 'Su']
        for i, d in enumerate(days):
            color = 7*16 if i < 5 else 10*16
            draw.text((x1+30+45*i, y1+8+(y2-y1)/5), d, font=font, anchor="mt", fill=color)
        font = ImageFont.truetype('./Open_Sans/OpenSans-SemiBold.ttf', int(radius*0.8))
        
        first_weekday, num_days_in_month = calendar.monthrange(x.year, x.month)
        deltay = ((y2-y1)*0.6)/((7 + num_days_in_month)/7)
        day = first_weekday
        for d in range(1, num_days_in_month+1):
            sx = x1+30+day*45
            sy = y1+deltay+((d+first_weekday-1)//7)*deltay+(y2-y1)/4
            if x.day == d:
                draw.ellipse((sx-20, sy-15, sx+20, sy+25), fill=9*16, width=20)
            color = 7*16 if day < 5 else 10*16
            color = 13*16 if x.day == d else color
            draw.text((sx, sy), str(d), font=font, anchor="mt", fill=color)
            day = day+1 if day+1 < 7 else 0
    
    def draw_clock(self, draw, xy, random=False):
        x = datetime.datetime.now().time()
        if random:
            x = datetime.time(hour=randint(1,12), minute=randint(1,59))
        hour = x.hour
        minute = x.minute

        (x1, y1, x2, y2) = xy
        # Circles and Shadow
        draw.ellipse((x1+4, y1+8, x2+4, y2+8), fill=13*16, outline=13*16, width=20)
        draw.ellipse((x1+2, y1+4, x2+2, y2+4), fill=11*16, outline=11*16, width=20)
        draw.ellipse((x1, y1, x2, y2), fill=13*16, outline=9*16, width=20)

        rx = (x2-x1)/2; ry = (y2-y1)/2;
        cx = x1 + rx; cy = y1 + ry;
        draw.ellipse((cx-8, cy-8, cx+8, cy+8), fill=13*16, outline=7*16, width=10)

        # Reference lines for time
        for i in range(0, 360, 30):
            sx = cx + (rx-35) * math.cos(i * math.pi / 180)
            sy = cy + (ry-35) * math.sin(i * math.pi / 180)
            ex = cx + (rx-20) * math.cos(i * math.pi / 180)
            ey = cy + (ry-20) * math.sin(i * math.pi / 180)
            draw.line((sx, sy, ex, ey), fill=7*16, width=5)
        
        # Hour
        hx = cx + (rx/2 * math.cos(((hour*30)+270) * math.pi / 180))
        hy = cy + (ry/2 * math.sin(((hour*30)+270) * math.pi / 180))
        draw.line((cx, cy, hx, hy), fill=6*16, width=7)

        # Minute
        mx = cx + (rx*3/4 * math.cos(((minute*6)+270) * math.pi / 180))
        my = cy + (ry*3/4 * math.sin(((minute*6)+270) * math.pi / 180))
        draw.line((cx, cy, mx, my), fill=8*16, width=5)

        font = ImageFont.truetype('./ds_digital/DS-DIGIB.TTF', 80)
        digital_time = x.strftime("%H:%M %p")
        draw.text(((x1+x2)/2, y2+50), digital_time, font=font, anchor="mm", fill=6*16)

    