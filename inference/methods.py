"""Load the original validation-selected matching rules and frozen checkpoints."""
import hashlib
import os
import pickle
from pathlib import Path

import numpy as np
import torch

from baselines import components
from classical_public import partial, cost, predict
from data_protocol import descriptors, features
from lean_inference import prepare_observations, tensor_inputs
from learners import Matcher, batch


def validate_file(path, digest):
    if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Unexpected SHA-256 for {path}")


class Inputs:
    """Share deterministic observed features across methods within one condition."""
    def __init__(self, sample):
        self.sample = sample
        self.cache = {}

    def once(self, key, prepare):
        if key not in self.cache:
            self.cache[key] = prepare()
        return self.cache[key]

    def features(self):
        return self.once("features", lambda: features(self.sample, True))

    def tensors(self, device):
        key = ("lean", device)
        return self.once(key, lambda: tensor_inputs(prepare_observations([self.sample]), device))

    def graph(self, device):
        def prepare():
            prepared = batch([self.features()], device, need_compat=True)
            prepared.pop("target")
            prepared.pop("reverse")
            return prepared
        return self.once(("graph", device), prepare)

    def descriptors(self):
        return self.once("descriptors", lambda: tuple(descriptors(self.sample[s]) for s in ("a", "b")))


class FrozenMatcher:
    def __init__(self, name, row, configuration, repo, inference, digests, device):
        self.name = name
        self.spec = row["spec"]
        self.family = self.spec["family"]
        self.threshold = row["threshold"]
        self.config = configuration["config"]
        self.device = device
        if self.family == "bcpd":
            binary = os.environ.get("PIGEON_BCPD_BIN", "")
            if not Path(binary).is_file() or not os.access(binary, os.X_OK):
                raise ValueError("Set PIGEON_BCPD_BIN to the executable built by environment/prepare_bcpd.py.")
        registered = self.spec["checkpoint"]
        if not registered:
            return
        if name == "joint_full-11":
            path = repo / "model" / registered
        else:
            path = inference / "weights" / Path(registered).name
        validate_file(path, digests[registered])
        if self.family == "tree":
            self.model = pickle.loads(path.read_bytes())
            if hasattr(self.model, "n_jobs"):
                self.model.n_jobs = 1
            return
        torch.manual_seed(2001)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if self.family == "physical":
            from physical_mixture import PhysicalMixture
            model = PhysicalMixture(**ck["config"])
        elif self.family == "kinematic":
            from kinematic_mixture import KinematicMixture
            model = KinematicMixture(**ck["config"])
        elif self.family == "twix":
            from coordinate_adapter import CoordinateTWiX
            model = CoordinateTWiX(self.spec["mode"], self.spec["layers"], True, device)
        elif self.family == "opgm":
            from opgm_adapter import MotionOPGM
            model = MotionOPGM(self.config["tau"])
        elif ck["kind"] == "temporal_graph":
            from temporal_models import TemporalMatcher
            model = TemporalMatcher(ck["pair_dim"], ck["node_dim"], ck["kind"])
        elif ck["kind"] == "pca_gm":
            from train_pca import PCAMatcher
            model = PCAMatcher(ck["node_dim"])
        elif ck["kind"] == "superglue":
            model = Matcher(ck["pair_dim"], ck["node_dim"], ck["kind"])
        else:
            raise ValueError(f"Unknown registered matcher {name}")
        model.load_state_dict(ck["state"], strict=True)
        self.model = model.to(device).eval()
        threshold_key = {"opgm": "rho", "twix": "threshold"}.get(self.family, "shift")
        if ck["validation"][threshold_key] != self.threshold:
            raise ValueError(f"Checkpoint and registry have different thresholds for {name}")

    def call(self, observed):
        s = observed.sample
        n, m = s["cost"].shape
        if self.family == "classical":
            kind = self.config["kind"]
            if kind == "relative_history":
                a, b = observed.descriptors()
                diff = np.clip((b["mu"][None] - a["mu"][:, None]) / 10., -20, 20).astype(np.float32)
                costs = np.log1p(np.sum(diff**2 * np.array(self.config["weights"]), 2) / self.config["scale"]**2)
            elif kind == "cv":
                costs = components(s)["cv"]
            elif kind == "kbest":
                return predict({"sample": s}, self.config), 0
            else:
                costs = cost(observed.features(), self.config)
            return partial(costs, self.threshold), 0
        if self.family == "bcpd":
            from bcpd_baseline import predict_cost
            a, b = observed.descriptors()
            costs, error = predict_cost({"sample": s, "node_a": a["node"], "node_b": b["node"]}, self.config)
            return partial(costs, self.threshold), int(error is not None)
        if self.family == "tree":
            pair = observed.features()["pair"].reshape(-1, 91)
            p = self.model.predict_proba(pair)[:, 1].clip(1e-6, 1-1e-6)
            return partial(-np.log(p/(1-p)).reshape(n, m), -self.threshold), 0
        if self.family == "twix":
            from twix_adapter import trajectory_input
            o = self.model(*trajectory_input({"sample": s}, self.device))
            return partial(-o.detach().cpu().numpy(), -self.threshold), 0
        if self.family == "opgm":
            x = observed.graph(self.device)
            if self.device == "cpu":
                # The author's feasibility tensor is used only by the training loss.
                # Its CUDA-only allocation does not enter the registered decoder.
                logits = self.model.backbone(x)["logits"]
                probability = self.model.sinkhorn(logits, x["ma"].sum(1), x["mb"].sum(1), dummy_row=True)
            else:
                o = self.model(x)
                logits, probability = o["affinity"], o["probability"]
            return self.model.decode(probability[0, :n, :m].cpu().numpy(),
                                     logits[0, :n, :m].cpu().numpy(), self.threshold), 0
        if self.name == "temporal_graph-11":
            x = observed.graph(self.device)
        else:
            x = observed.tensors(self.device)
            if self.name in ("pca_gm-11", "superglue-11"):
                x = {key: x[key] for key in ("a", "b", "ma", "mb")}
                if self.name == "superglue-11":
                    x["pair"] = torch.zeros((1, n, m, 1), device=self.device)
        o = self.model(x)
        logits = o["logits"][0, :n, :m] - .5*(o["rownull"][0, :n, None] + o["colnull"][0, None, :m])
        return partial(-logits.cpu().numpy(), -self.threshold), 0
