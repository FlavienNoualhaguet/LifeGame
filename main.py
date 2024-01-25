#!/usr/bin/env python3

from dataclasses import dataclass
import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from argparse import ArgumentParser

#----------------------------------------
#--------- GameLife Model ---------------
#----------------------------------------

@dataclass
class Cell:
    life:int

    @property
    def alive(self):
        return self.life == 1
    
    @property
    def dead(self):
        return not self.alive
    
    def born(self):
        self.life = 1
    
    def kill(self):
        self.life = 0

@dataclass
class Grid:
    n:int
    p:int

    def start(self):
        """ Create a random grid of 0 or 1 with shape (n, p)
            - 0 : dead
            - 1 : alive 
        """
        grid = np.zeros((self.n, self.p), dtype=int)
        for rd in np.random.randint(0, 2, int(self.n*self.p/4)):
            i, j = np.random.randint(0, self.n), np.random.randint(0, self.p)
            grid[i , j] = rd
        self.grid = np.vectorize(Cell)(life=grid)
        self.count_alive()
    
    def count_alive(self):
        self.nalive = np.sum([cell.alive for row in self.grid for cell in row])
        self.ndead  = self.n*self.p - self.nalive

    def count_neighbors(self, i, j):
        """ Count the number of alive neighbors around a cell at position (i, j) """
        count = 0
        for x in range(max(0, i - 1), min(self.n, i + 2)):
            for y in range(max(0, j - 1), min(self.p, j + 2)):
                if not (x == i and y == j) and self.grid[x, y].alive:
                    count += 1
        return count

    def evolve(self):
        """ Update the grid based on the rules of Conway's Game of Life """
        new_grid = np.empty((self.n, self.p), dtype=object)

        for i in range(self.n):
            for j in range(self.p):
                cell = self.grid[i, j]
                neighbors = self.count_neighbors(i, j)

                if cell.alive:
                    if neighbors < 2 or neighbors > 3: cell.kill()
                else:
                    if neighbors == 3: cell.born()

                new_grid[i, j] = cell

        self.grid = new_grid
        self.count_alive()

class AnimatedGrid(Grid):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.start()

    def animate(self, generations, interval, save_path=None):
        fig, ax = plt.subplots(figsize=(15,15))
        fig.suptitle(f"Grid of size {self.n} x {self.p}", fontsize=28, y=0.98, ha='center')
        
        def update(frame):
            if frame != 0:
                # Update and display the grid
                self.evolve()
            
            state_array = (np.vectorize(lambda cell: cell.alive)(self.grid)).astype(int)     
            
            ax.clear()
            im = ax.imshow(state_array, cmap='bwr', origin='upper')

            # for i in range(self.n + 1):
            #     ax.axhline(i - 0.5, color='black', linewidth=1)

            # for j in range(self.p + 1):
            #     ax.axvline(j - 0.5, color='black', linewidth=1)
            palive = 100.*self.nalive/(self.nalive+self.ndead)
            pdead  = 100 - palive
            axtile=f"""Generation {frame}/{generations}
Alive={self.nalive}/{self.n*self.p} ({palive:.2f}%) and Dead={self.ndead}/{self.n*self.p} ({pdead:.2f}%)"""
            ax.set_title(axtile, fontsize=24, pad=20)
            return im
        
        # Add colorbar with the same height as the axis
        cbar = plt.colorbar(update(0), ax=ax, fraction=0.046, pad=0.04)  # Adjust fraction and pad

        
        animation = FuncAnimation(fig, update, frames=generations, interval=interval, blit=False, repeat=False)
        
        if save_path:
            animation.save(save_path)  # Requires imagemagick to be installed
        else:
            plt.show()

#----------------------------------------
#------------- Main Part ----------------
#----------------------------------------
def parse_args():
    parser = ArgumentParser(description="Animate grid evolution",
                            prog="main")

    parser.add_argument("-n", type=int, default=50, help="Number of rows in the grid")
    parser.add_argument("-p", type=int, default=50, help="Number of columns in the grid")
    parser.add_argument("-g", "--generations", type=int, default=20, help="Number of generations")
    parser.add_argument("--interval", "-i", type=int, default=500, help="Time interval between frames in milliseconds")
    parser.add_argument("--save", "-s", default="animated_grid.gif", help="Filename to save the animated GIF")
    return parser.parse_args()




# Example usage
def main():
    args = parse_args()
    myAnimatedGrid  = AnimatedGrid(n=args.n, p=args.p)
    num_generations = args.generations
    update_interval = args.interval  # milliseconds
    gif_save_path   = args.save  

    myAnimatedGrid.animate(generations=num_generations, interval=update_interval, save_path=gif_save_path)

if __name__ == "__main__":
    main()