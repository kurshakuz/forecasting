import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from customer import Customer

class InspectionSpace:
    def __init__(self, x, y, width, height, frame_id, region_id):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.center = (x + width/2, y + width/2)
        self.frame_id = frame_id
        self.region_id = region_id

    def __repr__(self):
      return f"({str(self.frame_id)}-{str(self.region_id)})"

    def __print__(self):
      return f"({str(self.frame_id)}-{str(self.region_id)})"

def find_intersecting_rectangles(rectangles, squares):
    intersecting_rectangles = []

    for square in squares:
      square_left = square.x
      square_right = square.x + square.width
      square_top = square.y
      square_bottom = square.y + square.height
      # print(square_left, square_right, square_top, square_bottom)

      for rectangle in rectangles:
          rect_left = rectangle.x
          rect_right = rectangle.x + rectangle.width
          rect_top = rectangle.y
          rect_bottom = rectangle.y + rectangle.height

          if not (rect_right < square_left or rect_left > square_right or rect_bottom < square_top or rect_top > square_bottom):
                if rectangle not in intersecting_rectangles:
                    intersecting_rectangles.append(rectangle)

    return intersecting_rectangles

def generate_rectangles(num_regions, region_height_start, region_width, region_height):
    rectangles = []
    for i in range(num_regions*5):
        rectangle_x = i*region_width
        rectangle_y = region_height_start
        rectangle_width = region_width
        rectangle_height = region_height
        frame_id = i//num_regions
        region_id = i % num_regions
        rectangles.append(InspectionSpace(rectangle_x, rectangle_y, rectangle_width, rectangle_height, frame_id, region_id))

    return rectangles

def generate_hand_squares(hand_size, preds_pairs):
    squares = []
    for i, pair in enumerate(preds_pairs):
        pair = pair[0]
        x_l = pair[0]
        y_l = pair[1]
        x_r = pair[2]
        y_r = pair[3]
        square_height = hand_size

        if not (x_l==0.0 or y_l==0.0):
          square_x_l = 320*i + x_l-hand_size/2
          square_y_l = y_l-hand_size/2
          if square_x_l+hand_size > 320*(i+1):
            square_width_l = min(hand_size-0.01-(square_x_l+hand_size-320*(i+1)), hand_size)
          elif square_x_l < 320*i:
            square_width_l = 320*i - square_x_l
            square_x_l = 320*i + 0.01
          else:
            square_width_l = hand_size
          squares.append(InspectionSpace(square_x_l, square_y_l, square_width_l, square_height, -1, -1))

        if not (x_r==0.0 or y_r==0.0):
          square_x_r = 320*i + x_r-hand_size/2
          square_y_r = y_r-hand_size/2
          if square_x_r+hand_size > 320*(i+1):
            square_width_r = min(hand_size-0.01-(square_x_r+hand_size-320*(i+1)), hand_size)
          elif square_x_r < 320*i:
            square_width_r = 320*i - square_x_r
            square_x_r = 320*i + 0.01
          else:
            square_width_r = hand_size
          squares.append(InspectionSpace(square_x_r, square_y_r, square_width_r, square_height, -1, -1))

    return squares

def define_regions_and_intersections(frame_width, num_regions, reg_preds_pairs):
    region_height_start = 100
    region_height = 100
    region_width = frame_width / num_regions

    hand_size = 40
    squares = generate_hand_squares(hand_size, reg_preds_pairs)
    rectangles = generate_rectangles(num_regions, region_height_start, region_width, region_height)
    intersecting = find_intersecting_rectangles(rectangles, squares)

    return rectangles, squares, intersecting

def plot_hands(rectangles, squares, intersecting):
    img = np.zeros([320,320*5,3],dtype=np.uint8)
    img.fill(255)
    plt.figure(figsize=(15,10))
    imgplot = plt.imshow(img)
    plt.xlim([0, 1600])
    plt.ylim([320, 0])

    for rectangle in rectangles:
        # plot rectangles
        plt.gca().add_patch(Rectangle((rectangle.x, rectangle.y), rectangle.width, rectangle.height, linewidth=1, edgecolor='g', facecolor='none'))

    # line separators
    for i in range(4):
        plt.plot([320*(i+1), 320*(i+1)], [0, 320], color="black", linewidth=1)

    for square in squares:
      plt.gca().add_patch(Rectangle((square.x, square.y), square.width, square.height, linewidth=1, edgecolor='b', facecolor='none'))

    # Plotting the intersecting rectangles
    for rectangle in intersecting:
        intersect_rect = Rectangle((rectangle.x, rectangle.y), rectangle.width, rectangle.height, edgecolor='r', facecolor='none')
        plt.gca().add_patch(intersect_rect)

    # plt.xticks([])
    # plt.yticks([])
    plt.show()

def generate_tsptw_instances(frame_width, num_regions, rectangles, intersecting):
    time_scaler = 10000.0
    instances = []
    for i in range(5):
        id = 1
        # customers.append(Customer(id, point, rdy_time, due_date, serv_time))
        customers = []
        customers.append(Customer(id, (-40, 140), 0*time_scaler, 2*time_scaler, 0.0))
        id += 1
        for rectangle in rectangles[i*num_regions:(i+1)*num_regions]:
            # print(rectangle.center)
            center_x = rectangle.center[0] - rectangle.frame_id*frame_width
            center_y = rectangle.center[1]

            if rectangle in intersecting:
                # print(center_x)
                # print(rectangle)
                customers.append(Customer(id, (center_x, center_y), 0.5*time_scaler, 2*time_scaler, 0))
            else:
                customers.append(Customer(id, (center_x, center_y), 0*time_scaler, 0.5*time_scaler, 0.))
            id += 1
        instances.append(customers)
    return instances

def load_customers_from_predictions(preds):
    frame_width = 320
    num_regions = 4

    reg_preds = preds[:,:20]
    contact_pred = preds[:,20]
    reg_preds_pairs = np.split(reg_preds, 5, axis=1)

    rectangles, squares, intersecting = define_regions_and_intersections(frame_width, num_regions, reg_preds_pairs)
    instances = generate_tsptw_instances(frame_width, num_regions, rectangles, intersecting)

    return instances

if __name__ == "__main__":
    preds = np.array([[158.2681, 204.52983, 203.10486, 206.61021, 153.7534, 204.46507, 209.63512,
                        206.24173, 149.51596, 205.16443, 215.93907, 205.83598, 144.41257, 205.06213,
                        219.77278, 201.92041, 0., 0., 0., 0., 0.]])
    instances = load_customers_from_predictions(preds)
    print(instances[0])
