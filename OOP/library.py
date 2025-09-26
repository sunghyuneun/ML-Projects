class Library:

    def __init__(self):
        self.bookList = []

    def add_book(self, title, author):
        book = self.Book(title,author)
        self.bookList.append(book)
        return ("The book has been added.")

        
    def check_out_book(self, title):
        found = False
        for book in self.bookList:
            if (book.title == title):
                found = True
                if not book.is_checked_out:
                    book.is_checked_out = True
                    return "This book is now checked out."
                else:
                    return "This book is already checked out."
        if not found:
            return "This book was not found."

    def return_book(self, title):
        found = False
        for book in self.bookList:
            if (book.title == title):
                found = True
                if book.is_checked_out:
                    book.is_checked_out = False
                    return "This book is now returned."
                else:
                    return "This book was never checked out."
        if not found:
            return "This book was not found."

    def list_available_books(self):
        availableList = []
        for book in self.bookList:
            if not book.is_checked_out:
                availableList.append(book.title)
        return availableList

    class Book:
        def __init__(self, title, author):
            self.title = title
            self.author = author
            self.is_checked_out = False

        @property
        def title(self):
            return self._title
        
        @title.setter
        def title(self,title):
            self._title = title

        @property
        def author(self):
            return self._author
        
        @author.setter
        def author(self,author):
            self._author = author

        @property
        def is_checked_out(self):
            return self._is_checked_out
        
        @is_checked_out.setter
        def is_checked_out(self,is_checked_out):
            self._is_checked_out = is_checked_out

    

def main():
    pass

if __name__ == "__main__":
    main()