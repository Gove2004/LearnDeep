import torch
import torchsummary
import matplotlib.pyplot as plt

def run_learn(*args, **kwargs):
    """Simple Classification, Summary and Save Load"""
    
    MODEL_PATH = kwargs.get("learn4_model_path")
    JUST_PARMS = kwargs.get("learn4_just_parms", True)
    EPOCHS = 25
    LR = 0.1
    DEVICES = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[learn4] Using device: {DEVICES}")

    x0 = torch.normal(mean=1, std=1, size=(100, 2))
    x1 = torch.normal(mean=4, std=1, size=(100, 2))
    x = torch.cat((x0, x1), dim=0).to(DEVICES)
    y0 = torch.zeros(100)
    y1 = torch.ones(100)
    y = torch.cat((y0, y1), dim=0).long().to(DEVICES)

    def save_net(net, path, accuracy):
        if JUST_PARMS:
            torch.save(net.state_dict(), path)
            print(f"[learn4] Parameters saved to {path} with accuracy = {accuracy:.2f}")
        else:
            torch.save(net, path)
            print(f"[learn4] Model saved to {path} with accuracy = {accuracy:.2f}")

    def load_net(path):
        net = torch.nn.Sequential(
            torch.nn.Linear(2, 10),
            torch.nn.ReLU(),
            torch.nn.Linear(10, 2)
        ).to(DEVICES)

        use_checkpoint = input("[learn4] Load model from checkpoint? (y/n): ").strip().lower() == 'y'
        if not use_checkpoint:
            return net

        if JUST_PARMS:
            net.load_state_dict(torch.load(path))
            print(f"[learn4] Parameters loaded from {path}")
        else:
            net = torch.load(path)
            print(f"[learn4] Model loaded from {path}")
        return net

    net = load_net(MODEL_PATH)
    optimizer = torch.optim.SGD(net.parameters(), lr=LR)
    loss_func = torch.nn.CrossEntropyLoss()

    net_summary = torchsummary.summary(net, input_size=(2,), device=str(DEVICES))
    print(f"[learn4] Model Summary:\n{net_summary}")

    fig, ax = plt.subplots(figsize=(6, 6))
    best_accuracy = 0.0
    for epoch in range(EPOCHS):
        if not plt.fignum_exists(fig.number):
            print("[learn4] Window closed.")
            break

        prediction = net(x)
        loss = loss_func(prediction, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        prediction_classes = torch.argmax(prediction, dim=1)
        accuracy = (prediction_classes == y).float().mean().item()
        ax.cla()
        ax.scatter(x[:, 0].cpu().numpy(), x[:, 1].cpu().numpy(), c=prediction_classes.cpu().numpy(), cmap='coolwarm', alpha=0.5)
        ax.set_title(f'Epoch={epoch+1}, Loss={loss.item():.4f}, Accuracy={accuracy:.2f}')
        plt.pause(0.4)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            save_net(net, MODEL_PATH, accuracy)
    