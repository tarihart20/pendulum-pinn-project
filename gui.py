import math, tkinter as tk
from tkinter import ttk
import torch

from model.model import PendulumNN

G = 9.81

def load_model():
    model = PendulumNN()
    model.load_state_dict(torch.load("pendulum_nn.pt"))
    model.eval()    
    status = "Model loaded successfully."
    return model, status

class App:
    def __init__(self, root):
        self.root = root
        root.title("Simple Pendulum Simulator")
        root.geometry("960x650"); root.minsize(780, 540)
        self.model, status = load_model()
        self.length = tk.DoubleVar(value=.36)
        self.mass = tk.DoubleVar(value=.5)
        self.angle = tk.DoubleVar(value=25.)
        self.speed = tk.DoubleVar(value=1.)
        self.running = True; self.t = 0.; self.theta = math.radians(25); self.omega = 0.
        self.last_ms = None
        self.status = tk.StringVar(value=status)
        self._ui()
        root.after(16, self.tick)

    def _ui(self):
        outer = ttk.Frame(self.root, padding=12); outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Simple Pendulum Simulator", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(outer, text="Period estimate + numerical solution of the pendulum ODE").pack(anchor="w", pady=(0, 10))
        body = ttk.Frame(outer); body.pack(fill="both", expand=True)
        body.columnconfigure(0, weight=3); body.columnconfigure(1, weight=2); body.rowconfigure(0, weight=1)
        self.canvas = tk.Canvas(body, bg="#f5f8fc", highlightthickness=1, highlightbackground="#cbd5e1")
        self.canvas.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        panel = ttk.Frame(body); panel.grid(row=0, column=1, sticky="nsew")
        ttk.Label(panel, text="Controls", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self._scale(panel, "String length (m)", self.length, .15, 1.5, .01)
        self._scale(panel, "Bob mass (kg)", self.mass, .05, 5., .05)
        self._scale(panel, "Initial angle (degrees)", self.angle, 5, 60, 1)
        self._scale(panel, "Simulation speed", self.speed, .25, 2., .05)
        buttons = ttk.Frame(panel); buttons.pack(anchor="w", pady=8)
        ttk.Button(buttons, text="Pause / Resume", command=self.toggle).pack(side="left", padx=(0, 6))
        ttk.Button(buttons, text="Reset swing", command=self.reset).pack(side="left")
        ttk.Separator(panel).pack(fill="x", pady=8)
        ttk.Label(panel, text="Live results", font=("Segoe UI", 13, "bold")).pack(anchor="w")
        self.readout = ttk.Label(panel, text="", justify="left", font=("Segoe UI", 10))
        self.readout.pack(anchor="w", pady=8)
        ttk.Label(panel, text="For an ideal pendulum, mass does not change the period. Length does. The animation integrates θ'' = −(g/L) sin(θ).", wraplength=280).pack(anchor="w", pady=5)
        ttk.Label(outer, textvariable=self.status, foreground="#52677f", wraplength=900).pack(anchor="w", pady=(8, 0))

    def _scale(self, parent, label, var, lo, hi, step):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(8, 0))
        tk.Scale(parent, variable=var, from_=lo, to=hi, resolution=step, orient="horizontal",
                 length=270, showvalue=True, highlightthickness=0, bd=0,
                 command=lambda _v: self.refresh()).pack(fill="x")

    def learned_period(self):
        with torch.no_grad():
            return float(self.model(torch.tensor([[max(.001, self.length.get())]], dtype=torch.float32)).item())

    def refresh(self):
        L, m = self.length.get(), self.mass.get()
        T = 2*math.pi*math.sqrt(L/G)
        try: Tnn = self.learned_period()
        except Exception: Tnn = T
        E = m*G*L*(1-math.cos(math.radians(self.angle.get())))
        self.readout.config(text=f"NN period estimate: {Tnn:.3f} s\\nSmall-angle theory: {T:.3f} s\\nTheory frequency: {1/T:.3f} Hz\\nInitial potential energy: {E:.3f} J\\nSimulation time: {self.t:.2f} s\\nAngle now: {math.degrees(self.theta):.1f}°")

    def toggle(self): self.running = not self.running
    def reset(self):
        self.t = 0.; self.theta = math.radians(self.angle.get()); self.omega = 0.
        self.refresh()

    def deriv(self, th, om):
        return om, -(G/max(.01, self.length.get()))*math.sin(th)

    def integrate(self, dt):
        y, z = self.theta, self.omega
        a1,b1 = self.deriv(y,z)
        a2,b2 = self.deriv(y+dt*a1/2,z+dt*b1/2)
        a3,b3 = self.deriv(y+dt*a2/2,z+dt*b2/2)
        a4,b4 = self.deriv(y+dt*a3,z+dt*b3)
        self.theta += dt*(a1+2*a2+2*a3+a4)/6
        self.omega += dt*(b1+2*b2+2*b3+b4)/6
        self.t += dt

    def draw(self):
        c = self.canvas; c.delete("all")
        w, h = max(300,c.winfo_width()), max(300,c.winfo_height())
        px, py = w*.5, h*.17
        plen = max(55, min(h*.65,w*.6)*(self.length.get()/1.5))
        bx, by = px+plen*math.sin(self.theta), py+plen*math.cos(self.theta)
        r = 12+7*math.sqrt(self.mass.get()/5)
        c.create_text(16,14,anchor="nw",text=f"θ = {math.degrees(self.theta):.1f}°",fill="#26374a",font=("Segoe UI",11,"bold"))
        c.create_line(px-55,py,px+55,py,fill="#27384b",width=5)
        c.create_oval(px-6,py-6,px+6,py+6,fill="#27384b",outline="")
        c.create_line(px,py,bx,by,fill="#3d536c",width=3)
        c.create_oval(bx-r,by-r,bx+r,by+r,fill="#2878c8",outline="#17518b",width=2)
        c.create_text((px+bx)/2+10,(py+by)/2,text=f"{self.length.get():.2f} m",fill="#27384b")
        c.create_text(w/2,h-22,text="Ideal frictionless pendulum • gravity downward",fill="#52677f")

    def tick(self):
        now = int(self.root.tk.call("clock","milliseconds"))
        if self.last_ms is None: self.last_ms = now
        elapsed = min(.05, max(0., (now-self.last_ms)/1000)); self.last_ms = now
        if self.running:
            dt = elapsed*self.speed.get()
            n = max(1, int(math.ceil(dt/.004)))
            for _ in range(n): self.integrate(dt/n)
        self.draw(); self.refresh()
        self.root.after(16, self.tick)

def main():
    root = tk.Tk()
    try: ttk.Style().theme_use("clam")
    except tk.TclError: pass
    App(root); root.mainloop()

if __name__ == "__main__":
    main()