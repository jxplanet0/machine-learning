import pygame
import sys

# Inisialisasi Pygame
pygame.init()

# Warna dan ukuran
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
WIDTH, HEIGHT = 350, 550

# Inisialisasi layar Pygame
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Seven Segment Display")

# Konfigurasi segmen-segmen seven segment
segments = [
    # Segmen 0 (a)
    pygame.Rect(100, 50, 200, 20),
    # Segmen 1 (b)
    pygame.Rect(300, 70, 20, 200),
    # Segmen 2 (c)
    pygame.Rect(300, 300, 20, 200),
    # Segmen 3 (d)
    pygame.Rect(100, 500, 200, 20),
    # Segmen 4 (e)
    pygame.Rect(80, 300, 20, 200),
    # Segmen 5 (f)
    pygame.Rect(80, 70, 20, 200),
    # Segmen 6 (g)
    pygame.Rect(100, 280, 200, 20)
]


# Angka yang akan ditampilkan dalam seven segment
seven_segment_digits = {
    0: [1, 1, 1, 1, 1, 1, 0],
    1: [0, 1, 1, 0, 0, 0, 0],
    2: [1, 1, 0, 1, 1, 0, 1],
    3: [1, 1, 1, 1, 0, 0, 1],
    4: [0, 1, 1, 0, 0, 1, 1],
    5: [1, 0, 1, 1, 0, 1, 1],
    6: [1, 0, 1, 1, 1, 1, 1],
    7: [1, 1, 1, 0, 0, 0, 0],
    8: [1, 1, 1, 1, 1, 1, 1],
    9: [1, 1, 1, 1, 0, 1, 1]
}

def draw_seven_segment(number):
    segments_to_display = seven_segment_digits.get(number, [0, 0, 0, 0, 0, 0, 0])
    screen.fill(BLACK)

    for i in range(7):
        if segments_to_display[i]:
            pygame.draw.rect(screen, WHITE, segments[i])
        else:
            pygame.draw.rect(screen, BLACK, segments[i])

    pygame.display.flip()

# Loop utama
running = True
number_to_display = 0

# Mengatur batas laju pemutaran angka
clock = pygame.time.Clock()

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    draw_seven_segment(number_to_display)
    number_to_display = (number_to_display + 1) % 10

    # Delay 1 detik antara angka-angka yang ditampilkan
    pygame.time.delay(2000)

    # Perbarui layar dengan batas kecepatan tertentu
    clock.tick(1)

# Keluar dari aplikasi
pygame.quit()
sys.exit()
