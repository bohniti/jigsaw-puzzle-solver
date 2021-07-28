from langdon.core import init_step, load_step, log_step, train_step


def main():
    config, transform, model = init_step()
    train_dataloader, val_dataloader, test_dataloader = load_step(config, transform)
    tb_logger = log_step(config)
    train_step(config, model, train_dataloader, val_dataloader, test_dataloader, tb_logger)


if __name__ == "__main__":
    main()
